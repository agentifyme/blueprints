import openai
import re
import pandas as pd
from collections import defaultdict
import os
from dotenv import load_dotenv
import json
import logging

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def read_reviews(file_path):
    """
    Reads the review.txt file and extracts the product name and individual reviews.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract product name
    product_match = re.search(r'Product Name:\s*(.*?)\n\n', content, re.DOTALL)
    product_name = product_match.group(1).strip() if product_match else 'Unknown Product'

    # Extract reviews
    # This regex captures text between a number followed by ')' and the next number followed by ')' or end of string
    reviews = re.findall(r'\d+\)\s*(.*?)(?=\n\d+\)|$)', content, re.DOTALL)
    reviews = [review.strip().replace('\n', ' ') for review in reviews]

    return product_name, reviews

def extract_keywords_sentiment(review, max_retries=3):
    """
    Sends the review to OpenAI API and extracts keywords with their sentiment.
    Retries up to `max_retries` times if JSON parsing fails.
    """
    prompt = f"""
    Analyze the following product review and extract the key keywords along with their sentiment (Positive, Negative, Neutral).

    Review:
    "{review}"

    Please provide the results in the following exact JSON format without any additional text or comments:

    [
        {{"keyword": "example_keyword", "sentiment": "Positive"}},
        {{"keyword": "another_keyword", "sentiment": "Negative"}}
    ]
    """

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Use the correct model name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes product reviews for sentiment."},
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract the assistant's reply
            reply = response.choices[0].message['content'].strip()

            # Use regex to extract the JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', reply, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in the response.")

            reply_json = json_match.group(0)

            # Parse the JSON
            keywords = json.loads(reply_json)

            # Validate the structure
            if isinstance(keywords, list) and all('keyword' in item and 'sentiment' in item for item in keywords):
                return keywords
            else:
                raise ValueError("JSON structure is invalid.")

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} - Error processing review: {e}")
            logging.error(f"Review: {review}\nError: {e}\nResponse: {reply if 'reply' in locals() else 'No reply'}\n")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("Max retries reached. Skipping this review.")
                return []

    return []

def aggregate_keywords(all_keywords):
    """
    Aggregates keywords and counts their sentiment occurrences.
    """
    keyword_sentiment = defaultdict(lambda: {'Positive': 0, 'Negative': 0, 'Neutral': 0})

    for keywords in all_keywords:
        for item in keywords:
            keyword = item['keyword'].lower()
            sentiment = item['sentiment']
            if sentiment not in ['Positive', 'Negative', 'Neutral']:
                sentiment = 'Neutral'  # Default to Neutral if undefined
            keyword_sentiment[keyword][sentiment] += 1

    return keyword_sentiment

def main():
    # Path to your review.txt file
    file_path = 'reviews.txt'

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Please check the path.")
        return

    # Step 1: Read and parse the reviews
    product_name, reviews = read_reviews(file_path)
    print(f"Product Name: {product_name}")
    print(f"Number of Reviews: {len(reviews)}\n")

    all_keywords = []

    # Step 2: Process each review
    for idx, review in enumerate(reviews, 1):
        print(f"Processing Review {idx}/{len(reviews)}...")
        keywords = extract_keywords_sentiment(review)
        all_keywords.append(keywords)

    # Step 3: Aggregate the results
    aggregated = aggregate_keywords(all_keywords)

    # Convert to DataFrame for better visualization
    df = pd.DataFrame([
        {'Keyword': keyword, 'Positive': counts['Positive'], 'Negative': counts['Negative'], 'Neutral': counts['Neutral']}
        for keyword, counts in aggregated.items()
    ])

    # Sort by total mentions
    df['Total'] = df['Positive'] + df['Negative'] + df['Neutral']
    df = df.sort_values(by='Total', ascending=False).drop(columns=['Total'])

    # Step 4: Output the results
    print("\nAggregated Keyword Sentiment Analysis:")
    print(df)

if __name__ == "__main__":
    main()
