import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from transformers import pipeline

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
    """
    Resolves the channel ID from a video ID.
    """
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]
    except Exception as e:
        print(f"Error fetching channel ID from video: {e}")
    return None


def get_channel_from_username(username):
    """
    Resolves the channel ID from a username.
    """
    try:
        request = youtube.channels().list(part="id", forUsername=username)
        response = request.execute()
        if response["items"]:
            return response["items"][0]["id"]
    except Exception as e:
        print(f"Error fetching channel ID from username: {e}")
    return None


def get_channel_from_handle(handle):
    """
    Resolves the channel ID from a handle.
    """
    try:
        request = youtube.search().list(part="snippet", type="channel", q=f"@{handle}", maxResults=1)
        response = request.execute()
        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]
    except Exception as e:
        print(f"Error fetching channel ID from handle: {e}")
    return None


def fetch_channel_data(channel_id):
    """
    Fetches basic information about the channel.
    """
    try:
        request = youtube.channels().list(part="snippet,statistics", id=channel_id)
        response = request.execute()
        if response["items"]:
            channel = response["items"][0]
            title = channel["snippet"]["title"]
            description = channel["snippet"]["description"]
            subscribers = channel["statistics"]["subscriberCount"]
            views = channel["statistics"]["viewCount"]

            print(f"\nChannel Title: {title}")
            print(f"Description: {description}")
            print(f"Subscribers: {subscribers}")
            print(f"Total Views: {views}")
        else:
            print("Channel not found.")
    except Exception as e:
        print(f"Error fetching channel data: {e}")


def fetch_latest_video_details(channel_id):
    """
    Fetches the latest video details for a given channel ID.
    """
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

            print(f"\nLatest Video ID: {video_id}")
            print(f"Title: {title}")
            print(f"Description: {description}")
            print(f"Published At: {published_at}")

            # AI features: Summarize and analyze the description
            summarized_description = summarize_text(description)
            print(f"\nSummarized Description: {summarized_description}")

            sentiment = analyze_sentiment(description)
            print(f"Sentiment Analysis: {sentiment}")

            fetch_comments(video_id)
        else:
            print("No videos found for this channel.")
    except Exception as e:
        print(f"Error fetching latest video details: {e}")


def fetch_comments(video_id):
    """
    Fetches the top 5 comments for a given video ID.
    """
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=5
        )
        response = request.execute()
        if response["items"]:
            print("\nTop Comments:")
            for comment in response["items"]:
                author = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                print(f"{author}: {text}")

                # Summarize and analyze sentiment of the comment
                summarized_comment = summarize_text(text)
                print(f"Summarized Comment: {summarized_comment}")

                sentiment = analyze_sentiment(text)
                print(f"Sentiment Analysis for Comment: {sentiment}")
        else:
            print("No comments found for this video.")
    except Exception as e:
        print(f"Error fetching comments: {e}")


def summarize_text(text):
    """
    Summarizes the given text using the Hugging Face summarization pipeline.
    """
    try:
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None


def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using the Hugging Face sentiment-analysis pipeline.
    """
    try:
        sentiment = sentiment_analyzer(text)
        return sentiment[0]
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None


# Main function
def main():
    url = input("Enter the YouTube URL (channel, user, handle, custom, or video): ").strip()
    channel_id = resolve_channel_id(url)

    if not channel_id:
        print("Channel not found.")
        return

    fetch_channel_data(channel_id)
    fetch_latest_video_details(channel_id)


if __name__ == "__main__":
    main()
