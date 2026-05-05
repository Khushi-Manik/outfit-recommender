import "./App.css";
import { Routes, Route, Link } from "react-router-dom";
import {
  Facebook,
  Instagram,
  Palette,
  Play,
  Ruler,
  Shirt,
  Star,
  Sun,
  Twitter,
  UserRound,
} from "lucide-react";
import BodyTypeQuiz from "./pages/BodyTypeQuiz";
import NewsPage from "./pages/NewsPage";
import OutfitRecommendationsPage from "./pages/OutfitRecommendationsPage";
import ProductsPage from "./pages/ProductsPage";
import heroBoxImage from "./assets/hero-box-image.png";

const testimonialsData = [
  {
    name: "Aisha Khan",
    text: "Funky Fashion Finder has completely changed how I shop! The outfit recommendations are spot-on, and I finally understand what styles flatter my body type. Highly recommend!",
    rating: 5,
  },
  {
    name: "Raj Patel",
    text: "I used to struggle with putting outfits together. Now, with the wardrobe integration feature, I can easily create stylish looks from the clothes I already own. It's like having a personal stylist!",
    rating: 4,
  },
  {
    name: "Priya Sharma",
    text: "The body type analysis was so insightful! I've been dressing for the wrong shape my whole life. Now I feel more confident and fashionable. Thank you!",
    rating: 5,
  },
  {
    name: "Vikram Singh",
    text: "I love the playful twist of this app! It makes fashion fun and accessible. The recommendations are always unique and fit my personal style perfectly.",
    rating: 5,
  },
  {
    name: "Sneha Verma",
    text: "The Learn More section about different body types was incredibly helpful. I feel much more informed about fashion choices now.",
    rating: 4,
  },
];

function HomePage() {
  return (
    <>
      <section className="hero">
        <div className="hero-text">
          <h2>
            Discover Your <span className="gradient-text">Unique</span> Style With AI
          </h2>
          <p>
            Your personal fashion assistant with a playful twist. Get outfit recommendations,
            analyze your body type, and integrate your wardrobe.
          </p>
          <div className="buttons">
            <Link to="/recommendations" className="get-started">
              Get Started
            </Link>
            <Link to="/body-type" className="learn-more">
              Learn More
            </Link>
          </div>
        </div>
        <div className="hero-box">
          <img
            src={heroBoxImage}
            alt="Hero Illustration"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              borderRadius: "1rem",
            }}
          />
          <div className="label top-right">10k+ Happy Users</div>
          <div className="label bottom-left">95% Style Match Rate</div>
        </div>
      </section>

      <section className="features">
        <h2 className="section-title">Our Funky Features</h2>
        <div className="feature-grid">
          <Link to="/recommendations" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon" aria-hidden="true">
              <Shirt />
            </div>
            <h3>Outfit Recommendations</h3>
            <p>Get personalized outfit ideas based on your style preferences, occasion, and current trends.</p>
          </Link>
          <Link to="/body-type" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon" aria-hidden="true">
              <Ruler />
            </div>
            <h3>Body Type Analysis</h3>
            <p>Discover styles that flatter your unique body shape with our smart analysis tools.</p>
          </Link>
          <Link to="/fashion-news" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon" aria-hidden="true">
              <Palette />
            </div>
            <h3>Fashion News</h3>
            <p>Catch up on trends, sustainability, styling ideas, and fashion-tech updates in one place.</p>
          </Link>
        </div>
      </section>

      <section className="testimonials">
        <h2>What Our Users Say</h2>
        <p>Hear from people who have transformed their style with Funky Fashion Finder.</p>
        <div className="testimonial-grid">
          {testimonialsData.map((testimonial, index) => (
            <div key={index} className="testimonial-card">
              <div className="avatar" aria-hidden="true">
                <UserRound />
              </div>
              <div className="stars">
                {Array.from({ length: testimonial.rating }).map((_, i) => (
                  <span key={i} aria-hidden="true">
                    <Star fill="currentColor" />
                  </span>
                ))}
              </div>
              <p className="testimonial-text">"{testimonial.text}"</p>
              <p className="testimonial-author">- {testimonial.name}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="footer">
        <div className="footer-column">
          <h3>Funky Fashion Finder</h3>
          <p>Your playful guide to fashion recommendations and style inspiration.</p>
        </div>
        <div className="footer-column">
          <h4>Quick Links</h4>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/fashion-news">Fashion News</Link></li>
            <li><Link to="/products">Trending Products</Link></li>
            <li><Link to="/recommendations">Outfit Recommendations</Link></li>
          </ul>
        </div>
        <div className="footer-column">
          <h4>Connect With Us</h4>
          <div className="social-icons">
            <a href="#" aria-label="Instagram"><Instagram /></a>
            <a href="#" aria-label="Twitter"><Twitter /></a>
            <a href="#" aria-label="Facebook"><Facebook /></a>
            <a href="#" aria-label="YouTube"><Play /></a>
          </div>
        </div>
      </footer>

      <div className="copyright">
        Copyright 2025 Funky Fashion Finder. All rights reserved.
      </div>
    </>
  );
}

function App() {
  return (
    <div className="App">
      <header className="header">
        <h1 className="logo">
          <span>Funky </span>
          <span className="gradient-text">Fashion Finder</span>
        </h1>
        <nav className="nav-links">
          <Link to="/">Home</Link>
          <Link to="/fashion-news">Fashion News</Link>
          <Link to="/products">Trending Products</Link>
        </nav>
        <button type="button" className="sign-btn">
          <Sun aria-hidden="true" /> Sign Up
        </button>
      </header>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/body-type" element={<BodyTypeQuiz />} />
        <Route path="/fashion-news" element={<NewsPage />} />
        <Route path="/products" element={<ProductsPage />} />
        <Route path="/recommendations" element={<OutfitRecommendationsPage />} />
      </Routes>
    </div>
  );
}

export default App;
