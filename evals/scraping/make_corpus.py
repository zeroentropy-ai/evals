#!/usr/bin/env python3
"""
Script to scrape Airbnb help articles using r.jina.ai service.
Scrapes articles from https://www.airbnb.com/help/article/{id} for IDs 1-10000.
Uses r.jina.ai for enhanced web scraping capabilities.
"""

import os
import time
import requests
from pathlib import Path
import re
from typing import Optional, Dict, Any
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JinaWebScraper:
    """Web scraper using r.jina.ai service."""
    
    def __init__(self, output_dir: str = "./data/airbnb", delay: float = 0.2):
        """
        Initialize the Jina web scraper.
        
        Args:
            output_dir: Directory to save scraped articles
            delay: Delay between requests in seconds
        """
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        
        # Set up headers for r.jina.ai requests
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def count_existing_files(self, start_id: int = 1, end_id: int = 10000) -> int:
        """
        Count how many files already exist in the range.
        
        Args:
            start_id: Starting article ID
            end_id: Ending article ID
            
        Returns:
            Number of existing files
        """
        existing_count = 0
        for article_id in range(start_id, end_id + 1):
            file_path = self.output_dir / f"{article_id}.md"
            if file_path.exists():
                existing_count += 1
        return existing_count
        
    def scrape_with_jina(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a URL using r.jina.ai service.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with raw scraped content or None if failed
        """
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            logger.info(f"Scraping with Jina: {jina_url}")
            response = self.session.get(jina_url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Jina scraping failed with status code {response.status_code}")
                return None
            
            # Return raw content exactly as received from r.jina.ai
            raw_content = response.text
            
            if len(raw_content.strip()) < 50:  # Minimum content length
                logger.warning(f"Insufficient content from Jina for {url}")
                return None
            
            return {
                'raw_content': raw_content,
                'url': url,
                'source': 'r.jina.ai'
            }
            
        except requests.RequestException as e:
            logger.error(f"Error scraping with Jina {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping with Jina {url}: {e}")
            return None
    
    def scrape_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        """
        Scrape a single article by ID using r.jina.ai.
        
        Args:
            article_id: Article ID to scrape
            
        Returns:
            Dictionary with article data or None if article not found
        """
        url = f"https://www.airbnb.com/help/article/{article_id}"
        
        content_data = self.scrape_with_jina(url)
        
        if content_data:
            content_data['article_id'] = article_id
            return content_data
        else:
            return None
    
    def save_article(self, article_id: int, content_data: Dict[str, Any]) -> bool:
        """
        Save raw article content to file.
        
        Args:
            article_id: Article ID
            content_data: Article content data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            file_path = self.output_dir / f"{article_id}.md"
            
            # Save raw content exactly as received from r.jina.ai
            raw_content = content_data['raw_content']
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(raw_content)
            
            logger.info(f"Saved raw article {article_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving article {article_id}: {e}")
            return False
    
    def scrape_range(self, start_id: int = 1, end_id: int = 10000) -> None:
        """
        Scrape articles in a range of IDs using r.jina.ai.
        
        Args:
            start_id: Starting article ID
            end_id: Ending article ID
        """
        successful_scrapes = 0
        failed_scrapes = 0
        
        logger.info(f"Starting to scrape articles from {start_id} to {end_id} using r.jina.ai")
        
        for article_id in range(start_id, end_id + 1):
            content_data = self.scrape_article(article_id)
            
            if content_data:
                if self.save_article(article_id, content_data):
                    successful_scrapes += 1
                else:
                    failed_scrapes += 1
            else:
                failed_scrapes += 1
            
            # Progress update every 100 articles
            if article_id % 100 == 0:
                logger.info(f"Progress: {article_id}/{end_id} - Successful: {successful_scrapes}, Failed: {failed_scrapes}")
            
            # Delay between requests to be respectful
            time.sleep(self.delay)
        
        logger.info(f"Scraping completed. Successful: {successful_scrapes}, Failed: {failed_scrapes}")


def main():
    """Main function to run the Jina scraper."""
    scraper = JinaWebScraper(
        output_dir="./data/airbnb",
        delay= 3  # Reduced delay for faster scraping
    )
    
    # Check existing files
    #existing_count = scraper.count_existing_files(1, 10000)
    #logger.info(f"Found {existing_count} existing files out of 10000")
    
    # Test with a single article first
    logger.info("Testing with article 3225...")
    test_result = scraper.scrape_article(3225)
    if test_result:
        scraper.save_article(3225, test_result)
        logger.info("Test successful! Starting full scrape...")
        # Scrape articles 1-10000
        scraper.scrape_range(1, 10000)
    else:
        logger.error("Test failed! Check r.jina.ai service availability.")


if __name__ == "__main__":
    main()
