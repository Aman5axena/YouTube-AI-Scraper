import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from transformers import pipeline
from flask import Flask, request, render_template

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# Initialize AI features
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    revision="a4f8f3e"
)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"
)

# Flask App Setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        channel_id = resolve_channel_id(url)
        if channel_id:
            channel_data = fetch_channel_data(channel_id)
            video_data = fetch_latest_video_details(channel_id)
            return render_template("result.html", channel_data=channel_data, video_data=video_data)
        else:
            return render_template("index.html", error="Invalid YouTube URL")
    return render_template("index.html")

def resolve_channel_id(url):
    """
    Resolves the channel ID from the given YouTube URL.
    Supports channel, user, handle, and video URLs.
    """
    try:
        if "youtube.com/watch?v=" in url:  # Video URL
            video_id = url.split("v=")[-1].split("&")[0]
            return get_channel_from_video(video_id)
        elif "youtube.com/channel/" in url:  # Channel ID URL
            return url.split("/channel/")[-1].strip()
        elif "youtube.com/user/" in url:  # User URL
            username = url.split("/user/")[-1].strip()
            return get_channel_from_username(username)
        elif "youtube.com/@" in url:  # Handle URL
            handle = url.split("@")[-1].strip()
            return get_channel_from_handle(handle)
        else:
            raise ValueError("Invalid YouTube URL format.")
    except Exception as e:
        print(f"Error resolving channel ID: {e}")
        return None

def get_channel_from_video(video_id):
    """ Resolves the channel ID from a video ID. """
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]
    except Exception as e:
        print(f"Error fetching channel ID from video: {e}")
    return None

def get_channel_from_username(username):
    """ Resolves the channel ID from a username. """
    try:
        request = youtube.channels().list(part="id", forUsername=username)
        response = request.execute()
        if response["items"]:
            return response["items"][0]["id"]
    except Exception as e:
        print(f"Error fetching channel ID from username: {e}")
    return None

def get_channel_from_handle(handle):
    """ Resolves the channel ID from a handle. """
    try:
        request = youtube.search().list(part="snippet", type="channel", q=f"@{handle}", maxResults=1)
        response = request.execute()
        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]
    except Exception as e:
        print(f"Error fetching channel ID from handle: {e}")
    return None

def fetch_channel_data(channel_id):
    """ Fetches basic information about the channel. """
    try:
        request = youtube.channels().list(part="snippet,statistics", id=channel_id)
        response = request.execute()
        if response["items"]:
            channel = response["items"][0]
            title = channel["snippet"]["title"]
            description = channel["snippet"]["description"]
            subscribers = channel["statistics"]["subscriberCount"]
            views = channel["statistics"]["viewCount"]

            return {
                "title": title,
                "description": description,
                "subscribers": subscribers,
                "views": views
            }
        else:
            print("Channel not found.")
            return None
    except Exception as e:
        print(f"Error fetching channel data: {e}")
        return None

def fetch_latest_video_details(channel_id):
    """ Fetches the latest video details for a given channel ID. """
    try:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            order="date",
            maxResults=1
        )
        response = request.execute()
        if response["items"]:
            video = response["items"][0]
            video_id = video["id"]["videoId"]
            title = video["snippet"]["title"]
            description = video["snippet"]["description"]
            published_at = video["snippet"]["publishedAt"]

            summarized_description = summarize_text(description)
            sentiment = analyze_sentiment(description)

            # Fetch and analyze comments for the latest video
            comments = fetch_comments(video_id)

            return {
                "video_id": video_id,
                "title": title,
                "description": description,
                "published_at": published_at,
                "summarized_description": summarized_description,
                "sentiment": sentiment,
                "comments": comments  # Include comments in the result
            }
        else:
            print("No videos found for this channel.")
            return None
    except Exception as e:
        print(f"Error fetching latest video details: {e}")
        return None

def fetch_comments(video_id):
    """ Fetches the top 5 comments for a given video ID. """
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=5
        )
        response = request.execute()
        comments_data = []
        if response["items"]:
            for comment in response["items"]:
                author = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                
                # Summarize and analyze sentiment of the comment
                summarized_comment = summarize_text(text)
                sentiment = analyze_sentiment(text)

                comments_data.append({
                    "author": author,
                    "text": text,
                    "summarized_comment": summarized_comment,
                    "sentiment": sentiment
                })
        else:
            comments_data.append({"author": "No comments", "text": "No comments available."})
        
        return comments_data
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

def summarize_text(text):
    """ Summarizes the given text using the Hugging Face summarization pipeline. """
    try:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None

def analyze_sentiment(text):
    """ Analyzes the sentiment of the given text using the Hugging Face sentiment-analysis pipeline. """
    try:
        sentiment = sentiment_analyzer(text)
        return sentiment[0]
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None

if __name__ == "__main__":
    app.run(debug=True)
