import React from "react";
import "./App.css";
import { Routes, Route, Link } from "react-router-dom"; // Import Routes and Route
import BodyTypeQuiz from "./pages/BodyTypeQuiz";
import OutfitRecommendationsPage from "./pages/OutfitRecommendationsPage"; // Make sure this import is correct
import heroBoxImage from "./assets/hero-box-image.png"; // Adjust path if needed

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
    text: "The 'Learn More' section about different body types was incredibly helpful. I feel much more informed about fashion choices now.",
    rating: 4,
  },
];

function HomePage() {
  return (
    <>
      {/* Hero Section */}
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
            <button className="get-started">Get Started</button>
            <button className="learn-more">Learn More</button>
          </div>
        </div>
        <div className="hero-box">
          <img
            src={heroBoxImage}
            alt="Hero Illustration"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              borderRadius: '1rem',
            }}
          />
          <div className="label top-right">10k+ Happy Users</div>
          <div className="label bottom-left">95% Style Match Rate</div>
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <h2 className="section-title">Our Funky Features</h2>
        <div className="feature-grid">
          <Link to="/recommendations" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon">👕</div>
            <h3>Outfit Recommendations</h3>
            <p>Get personalized outfit ideas based on your style preferences, occasion, and current trends.</p>
          </Link>
          <Link to="/body-type" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon">📏</div>
            <h3>Body Type Analysis</h3>
            <p>Discover styles that flatter your unique body shape with our smart analysis tools.</p>
          </Link>
          <Link to="#" className="feature-card" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon">🎨</div>
            <h3>Wardrobe Integration</h3>
            <p>Connect your existing wardrobe and get suggestions on how to mix and match your clothes.</p>
          </Link>
        </div>
      </section>

      {/* Testimonials */}
      <section className="testimonials">
        <h2>What Our Users Say</h2>
        <p>Hear from people who have transformed their style with Funky Fashion Finder.</p>
        <div className="testimonial-grid">
          {testimonialsData.map((testimonial, index) => (
            <div key={index} className="testimonial-card">
              <div className="avatar">👤</div> {/* You can replace this with an actual user avatar */}
              <div className="stars">
                {Array.from({ length: testimonial.rating }).map((_, i) => (
                  <span key={i}>⭐</span>
                ))}
              </div>
              <p className="testimonial-text">"{testimonial.text}"</p>
              <p className="testimonial-author">- {testimonial.name}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-column">
          <h3>Funky Fashion Finder</h3>
          <p>Your playful guide to fashion recommendations and style inspiration.</p>
        </div>
        <div className="footer-column">
          <h4>Quick Links</h4>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="#">Fashion News</Link></li>
            <li><Link to="#">Trending Products</Link></li>
            <li><Link to="/recommendations">Outfit Recommendations</Link></li> {/* Keep this link */}
          </ul>
        </div>
        <div className="footer-column">
          <h4>Connect With Us</h4>
          <div className="social-icons">
            <a href="#">📸</a>
            <a href="#">🐦</a>
            <a href="#">📘</a>
            <a href="#">▶️</a>
          </div>
        </div>
      </footer>

      <div className="copyright">
        © 2025 Funky Fashion Finder. All rights reserved.
      </div>
    </>
  );
}

function App() {
  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <h1 className="logo">
          <span>Funky </span>
          <span className="gradient-text">Fashion Finder</span>
        </h1>
        <nav className="nav-links">
          <Link to="/">Home</Link>
          <Link to="#">Fashion News</Link>
          <Link to="#">Trending Products</Link>
        </nav>
        <button className="sign-btn">
          <span role="img" aria-label="sun">🌞</span> Sign Up
        </button>
      </header>

      {/* Routing */}
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/body-type" element={<BodyTypeQuiz />} />
        <Route path="/recommendations" element={<OutfitRecommendationsPage />} /> {/* Keep this route */}
      </Routes>
    </div>
  );
}

export default App;
