import praw
import pandas as pd
import re
import time
from datetime import datetime

# ------------------------------------------------------------
# 1. YOUR REDDIT API CREDENTIALS (GET THEM HERE: https://www.reddit.com/prefs/apps)
# ------------------------------------------------------------
# Click "Create App" -> Select "script" -> Set redirect uri to: http://localhost:8080
REDDIT_CLIENT_ID = "YOUR_CLIENT_ID"          # e.g., "XyZ123..."
REDDIT_CLIENT_SECRET = "YOUR_SECRET"         # e.g., "Abc456..."
REDDIT_USER_AGENT = "DeathBedScraper/0.1 (by u/YourUsername)"

# ------------------------------------------------------------
# 2. CATEGORY MAPPING (Subreddit -> DeathBed Category)
# ------------------------------------------------------------
# We pull hot posts from these subs and assign them the corresponding label.
SUBREDDIT_MAP = {
    # Career
    'careerguidance': 'career',
    'jobs': 'career',
    'cscareerquestions': 'career',
    'financialcareers': 'career',
    
    # Education
    'gradadmissions': 'education',
    'college': 'education',
    'ApplyingToCollege': 'education',
    'education': 'education',
    
    # Relationship
    'relationship_advice': 'relationship',
    'dating_advice': 'relationship',
    'marriage': 'relationship',
    
    # Finance
    'personalfinance': 'finance',
    'investing': 'finance',
    'FinancialPlanning': 'finance',
    
    # Relocation
    'moving': 'relocation',
    'expats': 'relocation',
    'IWantOut': 'relocation',
    
    # Family
    'parenting': 'family',
    'family': 'family',
    'NewParents': 'family',
    
    # Health
    'health': 'health',
    'mentalhealth': 'health',
    'Fitness': 'health',
    
    # Lifestyle
    'minimalism': 'lifestyle',
    'decidingtobebetter': 'lifestyle',
    'simpleliving': 'lifestyle',
}

# ------------------------------------------------------------
# 3. TEXT CLEANING FUNCTION
# ------------------------------------------------------------
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ------------------------------------------------------------
# 4. MAIN SCRAPING LOGIC
# ------------------------------------------------------------
def scrape_reddit(limit_per_sub=250):
    """
    Scrapes hot posts from each subreddit.
    limit_per_sub: How many posts to pull per subreddit (max ~1000).
    """
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    data = []
    
    for subreddit_name, category in SUBREDDIT_MAP.items():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scraping r/{subreddit_name} -> {category}")
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Use 'hot' for general advice, you can switch to 'new' or 'top'
            for post in subreddit.hot(limit=limit_per_sub):
                
                # ---- Filter: Only keep posts that look like decisions ----
                title_lower = post.title.lower()
                
                # Must contain a question or "should I" or "deciding between"
                is_decision = (
                    '?' in title_lower or 
                    'should i' in title_lower or 
                    'deciding between' in title_lower or
                    'what should' in title_lower or
                    'is it worth' in title_lower
                )
                
                if not is_decision:
                    continue
                
                # Combine Title + First 300 chars of Selftext
                combined_text = post.title + " " + (post.selftext[:300] if post.selftext else "")
                
                # Clean it
                cleaned = clean_text(combined_text)
                
                # Skip if too short after cleaning
                if len(cleaned) < 20:
                    continue
                
                data.append({
                    'text': cleaned,
                    'label': category,
                    'source_subreddit': subreddit_name,
                    'score': post.score,      # Keep for potential filtering later
                    'created_utc': post.created_utc
                })
                
        except Exception as e:
            print(f"  [!] Error scraping r/{subreddit_name}: {e}")
            continue
        
        # Be nice to Reddit's API (rate limit)
        time.sleep(2)
    
    return pd.DataFrame(data)

# ------------------------------------------------------------
# 5. RUN IT
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting DeathBed Reddit Scraper...")
    
    df = scrape_reddit(limit_per_sub=250)
    
    # Remove duplicates (exact same text)
    df = df.drop_duplicates(subset=['text'])
    
    # Optional: Filter out posts with low score (to keep quality high)
    # df = df[df['score'] >= 5]
    
    # Save to CSV
    df.to_csv('data/raw/reddit_raw.csv', index=False)
    
    print(f"✅ Scraping Complete! Collected {len(df)} unique decision posts.")
    print(f"📊 Category Distribution:\n{df['label'].value_counts()}")