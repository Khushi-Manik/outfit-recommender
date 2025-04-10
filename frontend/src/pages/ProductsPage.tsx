import React from 'react';
import './ProductsPage.css'; // Import the CSS
import funkyShirtImage from '@/assets/funky-shirt.avif';
import neonSneakersImage from '@/assets/neon-sneakers.avif';
import retroHatImage from '@/assets/retro-hat.jpg';
import graphicTeeImage from '@/assets/graphic-tee.avif';
import miniSkirtImage from  '@/assets/mini-skirt.avif';
import chunkyNecklaceImage from  '@/assets/chunky-necklace.avif';
import cardiganImage from  '@/assets/cardigan.avif';
import bagImage from '@/assets/bag.avif';
import linenPantsImage from '@/assets/linen-pants.avif';
import sunglassesImage from '@/assets/sunglasses.avif';


interface Product {
  id: number;
  name: string;
  price: number; // Price will now be in Rupees
  image: string;
  description?: string;
  discount?: number; // Discount still as a percentage
  category?: string;
  rating?: number;
  link?: string; // To link to a detailed product page
}

const ProductsPage: React.FC = () => {
  // In a real app, this data would come from an API or CMS
  const products: Product[] = [
    {
      id: 1,
      name: 'Funky Patterned Shirt',
      price: 354,
      image: funkyShirtImage,
      description: 'A vibrant shirt with a unique pattern to make you stand out.',
      category: 'Tops',
      rating: 4.5,
      link: 'https://www.ajio.com/search/?text=Funky%20Patterned%20Shirt%20for%20women',
    },
    {
      id: 2,
      name: 'Neon Platform Sneakers',
      price: 1176,
      image: neonSneakersImage,
      description: 'Step up your shoe game with these bold and bright platform sneakers.',
      discount: 15,
      category: 'Shoes',
      rating: 4.2,
      link: 'https://www.ajio.com/search/?text=Neon%20Platform%20Sneakers',
    },
    {
      id: 3,
      name: 'Retro Bucket Hat',
      price: 1495,
      image: retroHatImage,
      description: 'Add a touch of vintage style to your look with this classic bucket hat.',
      category: 'Accessories',
      rating: 4.8,
      link: 'https://www.ajio.com/supervek-tie--dye-reversible-bucket-hat/p/466615804_multi',
    },
    {
      id: 4,
      name: 'Oversized Graphic Tee',
      price: 999,
      image: graphicTeeImage,
      description: 'A comfortable and stylish oversized tee featuring a cool graphic.',
      category: 'Tops',
      rating: 4.0,
      link: 'https://www.ajio.com/h-m-women-oversized-graphic-t-shirt/p/700414456_white',
    },
    {
      id: 5,
      name: 'Y2K Mini Skirt',
      price: 549,
      image: miniSkirtImage,
      description: 'Bring back the early 2000s with this trendy denim mini skirt.',
      category: 'Bottoms',
      rating: 3.9,
      link: '/product/y2k-skirt',
    },
    {
      id: 6,
      name: 'Chunky Statement Necklace',
      price: 999,
      image: chunkyNecklaceImage,
      description: 'Make a bold statement with this eye-catching chunky necklace.',
      category: 'Accessories',
      rating: 4.6,
      link: '/product/chunky-necklace',
    },
    {
      id: 7,
      name: 'Color Block Cardigan',
      price: 2661,
      image: cardiganImage,
      description: 'Stay cozy and fashionable in this stylish color block cardigan.',
      discount: 20,
      category: 'Outerwear',
      rating: 4.3,
      link: '/product/color-cardigan',
    },
    {
      id: 8,
      name: 'Holographic Crossbody Bag',
      price: 4549,
      image: bagImage,
      description: 'Carry your essentials in style with this futuristic holographic crossbody bag.',
      category: 'Bags',
      rating: 4.1,
      link: '/product/holographic-bag',
    },
    {
      id: 9,
      name: 'Striped Linen Pants',
      price: 2699,
      image: linenPantsImage,
      description: 'Lightweight and breathable linen pants perfect for warm weather.',
      category: 'Bottoms',
      rating: 4.4,
      link: '/product/linen-pants',
    },
    {
      id: 10,
      name: 'Aviator Sunglasses',
      price: 8090,
      image: sunglassesImage,
      description: 'Classic aviator sunglasses that never go out of style.',
      category: 'Accessories',
      rating: 4.7,
      link: '/product/aviator-sunglasses',
    },
  ];

  return (
    <div className="products-page-container">
      <h1 className="products-page-title">Trending Products</h1>
      <ul className="product-list">
        {products.map((product) => (
          <li key={product.id} className="product-item">
            <div className="product-image-container">
              <img src={product.image || '/images/placeholder.svg'} alt={product.name} className="product-image" />
            </div>
            <div className="product-details">
              <h3 className="product-name">{product.name}</h3>
              <p className="product-price">
                {product.discount && (
                  <span className="original-price">₹{product.price.toFixed(2)}</span>
                )}
                ₹{(product.discount ? (product.price * (1 - product.discount / 100)).toFixed(2) : product.price.toFixed(2))}
                {product.discount && <span className="discount-badge">-{product.discount}%</span>}
              </p>
              {product.description && <p className="product-description">{product.description.substring(0, 80)}...</p>}
              <div className="product-meta">
                {product.category && <span className="product-category">{product.category}</span>}
                {product.rating && (
                  <span className="product-rating">
                    <svg className="star-icon" viewBox="0 0 24 24">
                      <path d="M12 17.27L18.18 21L16.54 13.97L22 9.24L14.81 8.63L12 2L9.19 8.63L2 9.24L7.46 13.97L5.82 21L12 17.27Z" fill="#ffc107"/>
                    </svg>
                    {product.rating.toFixed(1)}
                  </span>
                )}
              </div>
              <div className="product-actions">
                {product.link && (
                  <a href={product.link} className="view-details-button">View Details</a>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ProductsPage;