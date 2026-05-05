import React from 'react';
import './NewsPage.css'; // Import the CSS file

import sustainableFashionImage from '@/assets/cardigan.avif';
import springSummerTrendsImage from '@/assets/graphic-tee.avif';
import aiStylingImage from '@/assets/hero-box-image.png';
import classicDenimImage from '@/assets/linen-pants.avif';

interface NewsArticle {
  id: number;
  title: string;
  date: string;
  excerpt: string;
  imageUrl?: string;
  link: string;
}

const featuredNews: NewsArticle[] = [
  {
    id: 1,
    title: 'The Rise of Sustainable Fashion in 2025',
    date: 'April 5, 2025',
    excerpt: 'Discover the latest trends in eco-conscious clothing and how the fashion industry is embracing sustainability.',
    imageUrl: sustainableFashionImage, // Use the imported image
    link: 'https://ethikonline.com/blogs/featured-articles/sustainable-fashion-trends-2025#:~:text=What%20are%20the%20key%20sustainable,a%20minimalist%20approach%20to%20consumption.',
  },
];

const recentNews: NewsArticle[] = [
  {
    id: 2,
    title: 'Top Style Trends for Spring/Summer 2025',
    date: 'April 3, 2025',
    excerpt: 'From bold colors to relaxed silhouettes, get a head start on the must-have styles for the upcoming season.',
    imageUrl: springSummerTrendsImage, // Use the imported image
    link: 'https://www.cosmopolitan.com/style-beauty/fashion/a64272587/spring-2025-fashion-trends/',
  },
  {
    id: 3,
    title: 'How AI is Revolutionizing Personal Styling',
    date: 'April 1, 2025',
    excerpt: 'Explore the innovative ways artificial intelligence is transforming how we discover and shop for fashion.',
    imageUrl: aiStylingImage, // Use the imported image
    link: 'https://www.lefashionpost.com/2024/10/28/how-ai-is-revolutionizing-your-fashion-and-beauty-experience/',
  },
  {
    id: 4,
    title: 'The Timeless Appeal of Classic Denim',
    date: 'March 30, 2025',
    excerpt: 'A look back at the enduring popularity of denim and how to style it for a modern look.',
    imageUrl: classicDenimImage, // Use the imported image
    link: 'https://apparelresources.com/fashion-news/trends/s-s-25-denim-innovations-washes-finishes-treatments/',
  },
];

const categories = ['Trends', 'Style Guides', 'Sustainability', 'Celebrity Fashion', 'Shopping'];

const NewsPage: React.FC = () => {
  const handleNewsletterSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
  };

  return (
    <div className="news-page-container">
      <header className="news-header">
        <h1>Fashion News & Trends</h1>
        <p className="subheader">Stay up-to-date with the latest happenings in the world of fashion.</p>
      </header>

      <section className="featured-news">
        <h2>Featured Story</h2>
        {featuredNews.map((article) => (
          <div key={article.id} className="featured-article">
            {article.imageUrl && <img src={article.imageUrl} alt={article.title} className="featured-image" />}
            <div className="featured-content">
              <h3><a href={article.link} target="_blank" rel="noreferrer">{article.title}</a></h3>
              <p className="article-date">{article.date}</p>
              <p className="article-excerpt">{article.excerpt}</p>
              <a href={article.link} className="read-more" target="_blank" rel="noreferrer">Read More</a>
            </div>
          </div>
        ))}
      </section>

      <section className="recent-news">
        <h2>Recent Articles</h2>
        <div className="recent-articles-grid">
          {recentNews.map((article) => (
            <div key={article.id} className="recent-article-card">
              {article.imageUrl && <img src={article.imageUrl} alt={article.title} className="article-image" />}
              <div className="article-details">
                <h3><a href={article.link} target="_blank" rel="noreferrer">{article.title}</a></h3>
                <p className="article-date">{article.date}</p>
                <p className="article-excerpt">{article.excerpt.substring(0, 100)}...</p>
                <a href={article.link} className="read-more" target="_blank" rel="noreferrer">Read More</a>
              </div>
            </div>
          ))}
        </div>
      </section>

      <aside className="news-sidebar">
        <div className="categories">
          <h3>Categories</h3>
          <ul>
            {categories.map((category) => (
              <li key={category}><span>{category}</span></li>
            ))}
          </ul>
        </div>

        <div className="newsletter">
          <h3>Subscribe to Our Newsletter</h3>
          <p>Get the latest fashion news delivered to your inbox.</p>
          <form onSubmit={handleNewsletterSubmit}>
            <input type="email" placeholder="Your Email Address" />
            <button type="submit">Subscribe</button>
          </form>
        </div>
      </aside>
    </div>
  );
};

export default NewsPage;
