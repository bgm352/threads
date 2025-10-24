import streamlit as st
import requests
import pandas as pd
import json
import threading
import time

# Dependency checks
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Basic app setup
st.set_page_config(page_title="Threads DOL Finder", layout="wide", page_icon="🧵")
st.title("🧵 Threads DOL & Profile Vetting — GPT-5 + Gemini 3 Support")

# Runs a specified Apify actor with a payload, returns dataset items
def run_apify_actor(actor_id, api_key, payload):
    run_url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
    headers = {"Authorization": f"Bearer {api_key}"}

    resp = requests.post(run_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    run_id = resp.json().get("data", {}).get("id")
    if not run_id:
        raise RuntimeError("No run ID from Apify")

    # Poll with timeout for run to complete
    for _ in range(60):
        status = requests.get(f"{run_url}/{run_id}", headers=headers).json().get("data", {}).get("status")
        if status == "SUCCEEDED":
            break
        time.sleep(5)
    else:
        raise TimeoutError("Apify actor run timed out")

    dataset_id = requests.get(f"{run_url}/{run_id}", headers=headers).json()["data"]["defaultDatasetId"]
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true"
    dataset_resp = requests.get(url, headers=headers, timeout=60)
    dataset_resp.raise_for_status()
    return dataset_resp.json()

@st.cache_data(show_spinner=True)
def scrape_threads_posts(api_key, keywords, max_items):
    posts = []
    for kw in keywords:
        try:
            # futurizerush/meta-threads-scraper for search posts
            search_posts = run_apify_actor(
                "futurizerush~meta-threads-scraper",
                api_key,
                {"search": kw, "maxItems": max_items, "includeComments": False}
            )
            # curious_coder/threads-scraper for detailed posts
            detailed_posts = run_apify_actor(
                "curious_coder~threads-scraper",
                api_key,
                {"queries": [kw], "maxItems": max_items, "proxyCountryCode": "US"}
            )
            combined = []
            for p in search_posts + detailed_posts:
                user = p.get("user", {})
                combined.append({
                    "keyword": kw,
                    "username": user.get("username"),
                    "verified": user.get("is_verified"),
                    "post_text": p.get("caption", {}).get("text", ""),
                    "likes": p.get("like_count", 0),
                    "replies": p.get("reply_count", 0),
                    "timestamp": p.get("taken_at")
                })
            posts.extend(combined)
            time.sleep(3)  # Gentle pause
        except Exception as e:
            st.warning(f"Error scraping '{kw}': {e}")
            continue
    return pd.DataFrame(posts)

@st.cache_data(show_spinner=True)
def enrich_threads_profiles(api_key, usernames):
    enriched = []
    for u in usernames[:30]:  # limit batch for demo
        try:
            prof_data = run_apify_actor(
                "apify~threads-profile-api-scraper",
                api_key,
                {"profiles": [u]}
            )
            for p in prof_data:
                enriched.append({
                    "username": p.get("username"),
                    "followers": p.get("follower_count"),
                    "bio": p.get("biography", ""),
                    "full_name": p.get("full_name"),
                    "verified": p.get("is_verified")
                })
        except Exception as e:
            st.warning(f"Profile enrichment failed for {u}: {e}")
    return pd.DataFrame(enriched)

def ai_vet_threads(data, model, provider, api_key, temperature=0.4):
    prompt = (
        "Analyze these Threads users and posts for influence as Digital Opinion Leaders. "
        "Return JSON list with fields: username, influence_score (0-1), is_DOL (true/false), credibility ('high','medium','low'), and topic summary.\n\n"
        f"{json.dumps(data[:5], indent=2)}"
    )
    try:
        if provider == "OpenAI":
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
            )
            output = resp.choices[0].message["content"]
        else:
            if not genai:
                raise RuntimeError("Gemini SDK not installed.")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            output = model_obj.generate_content(prompt).text

        return pd.DataFrame(json.loads(output))
    except Exception as e:
        st.error(f"AI vetting failed: {e}")
        return pd.DataFrame()

def threaded(func, *args):
    res = {}
    def wrapper():
        try:
            res["data"] = func(*args)
        except Exception as err:
            res["error"] = str(err)
    t = threading.Thread(target=wrapper)
    t.start()
    t.join()
    if "error" in res:
        raise RuntimeError(res["error"])
    return res.get("data")

# Sidebar
st.sidebar.header("Setup APIs and Model")

apify_key = st.sidebar.text_input("Apify API Token", type="password")
provider = st.sidebar.selectbox("AI Provider", ["OpenAI GPT", "Google Gemini"])
if provider == "OpenAI GPT":
    ai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("OpenAI Model", ["gpt-5-pro", "gpt-4-turbo", "gpt-3.5-turbo"])
else:
    ai_key = st.sidebar.text_input("Gemini API Key", type="password")
    model = st.sidebar.selectbox("Gemini Model", ["gemini-3.0-pro", "gemini-2.5-pro"])

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.5)

st.sidebar.header("Search Parameters")
keyword_input = st.sidebar.text_area("Enter Topics / Keywords (one per line)")
keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
max_posts = st.sidebar.number_input("Max Posts per Keyword", 10, 200, 50)

# Main buttons & logic
if st.button("🚀 Scrape Threads Posts"):
    if not apify_key:
        st.warning("Enter your Apify API token")
    elif not keywords:
        st.warning("Enter at least one keyword")
    else:
        with st.spinner("Scraping posts..."):
            scraped_df = threaded(scrape_threads_posts, apify_key, keywords, max_posts)
            if not scraped_df.empty:
                st.session_state["threads_posts"] = scraped_df
                st.success(f"Scraped {len(scraped_df)} posts")
                st.dataframe(scraped_df.head(20))
            else:
                st.error("No posts found")

if st.button("🧩 Enrich Profiles"):
    posts_df = st.session_state.get("threads_posts", pd.DataFrame())
    if posts_df.empty:
        st.warning("Run scraper first")
    else:
        with st.spinner("Enriching profile data..."):
            usernames = posts_df["username"].dropna().unique().tolist()
            profiles_df = threaded(enrich_threads_profiles, apify_key, usernames)
            if not profiles_df.empty:
                merged_df = posts_df.merge(profiles_df, on="username", how="left")
                st.session_state["enriched_data"] = merged_df
                st.success("Profiles enriched")
                st.dataframe(merged_df.head(20))
            else:
                st.error("Profile enrichment returned no data")

if st.button("🤖 AI Vet Digital Opinion Leaders"):
    data_df = st.session_state.get("enriched_data", pd.DataFrame())
    if data_df.empty:
        st.warning("Need scraped & enriched data")
    elif not ai_key:
        st.warning(f"Enter your {provider} API key")
    else:
        with st.spinner("Running AI vetting..."):
            vetted_df = threaded(ai_vet_threads, data_df.to_dict("records"), model, provider.split()[0], ai_key, temperature)
            if not vetted_df.empty:
                st.session_state["vetted_df"] = vetted_df
                st.success("AI vetting complete")
                st.dataframe(vetted_df)
                st.download_button("Download Vetting Results CSV", vetted_df.to_csv(index=False), "vetted_threads.csv")
            else:
                st.error("AI vetting returned no results")
