import React, { useState } from 'react';
import './FashionAssistant.css';

const FashionAssistant: React.FC = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setResponse('');
  
    // Simulate API delay
    setTimeout(() => {
        const mockResponses: Record<string, string> = {
            // 👋 Casual conversation
            "hi": "Hey there! 👋 Ready to slay today?",
            "hello": "Hello, fashion star! ✨ What are you in the mood to wear?",
            "hey": "Hey hey! Need some style inspo?",
            "how are you": "Feeling fab and fashionable! How can I help you today?",
            "who are you": "I’m your personal fashion assistant 💅 — here to help you look your best, always.",
            "what can you do": "I can recommend outfits, suggest colors, guide styles for events, and help you level up your look!"
          
            // 👗 Outfit recommendations
            , "what should i wear to a wedding": "A flowy pastel dress or a tailored suit works beautifully. Add elegant accessories and you're golden. 💍"
            , "what should i wear on a first date": "Something comfy but confident — maybe dark jeans with a nice shirt or a casual dress with cute sneakers. 💖"
            , "what should a guy wear to a wedding": "A fitted blazer, dress shirt, chinos or dress pants, and loafers. Add a pocket square for flair!"
            , "what should a girl wear to college": "Keep it casual and comfy — jeans, graphic tees, oversized hoodies, or simple dresses with sneakers. 🎒"
            , "what should i wear for an interview": "A clean, professional look: button-up shirt, blazer, and dress pants or a pencil skirt. Keep accessories minimal. ✅"
            , "what can i wear to a party": "Try a bold outfit — crop top and skirt combo, jumpsuit, or a sleek shirt with trousers. Add statement jewelry!"
            , "what should i wear to the gym": "Moisture-wicking clothes like a tank and leggings or shorts and a T-shirt. Don’t forget comfy shoes! 💪"
            , "what to wear at home": "Comfy loungewear — think joggers, oversized tees, hoodies, or cute pajama sets!"
          
            // ☀️🌧️ Weather-based
            , "what should i wear in winter": "Stay warm! Try a layered look: thermal top, sweater, long coat, boots, and a chunky scarf. ❄️"
            , "what should i wear in summer": "Think breezy and light — tank tops, cotton shorts, sundresses, sandals, and lots of water. ☀️"
            , "what should i wear in rainy weather": "Waterproof jacket, boots, and an umbrella are musts. Go with quick-dry clothes to stay comfy. 🌧️"
            , "what to wear in spring": "Floral prints, cardigans, light jeans, and sneakers — spring is all about colors and layers!"
          
            // 🎨 Skin tone & body type
            , "what colors suit fair skin": "Try soft pastels, jewel tones, and navy blue. Avoid colors that wash you out like pale yellow."
            , "what colors suit dark skin": "Bold, bright, and vibrant tones like red, orange, cobalt blue, and white look 🔥 on dark skin."
            , "how should i dress for my body type": "Each body is beautiful! Let me know your type and I’ll suggest cuts that flatter you best."
            , "what suits a pear shaped body": "Highlight your top half! Try boat neck tops, flared skirts, and avoid tight bottoms."
            , "what suits an apple body shape": "Empire waist dresses and flowy tops work great. Avoid clingy fabrics around the waist."
            , "what suits a rectangular body": "Create curves with peplum tops, ruffled blouses, and belted outfits!"
          
            // 👟 Styling questions
            , "can i wear sneakers with a dress": "Yes! It's a trendy and comfy combo that adds a casual twist to any outfit. 🏃‍♀️"
            , "how do i layer clothes": "Start with a base layer (like a tee), add a mid layer (sweater or hoodie), and finish with a jacket or coat. Mix textures!"
            , "how do i style baggy jeans": "Pair them with a fitted top, crop top, or tucked-in shirt. Add sneakers or boots to complete the vibe."
            , "can i mix patterns": "Absolutely — just stick to a common color palette and mix patterns of different scales (like stripes with florals)."
            , "what accessories should i wear": "Try layered necklaces, hoop earrings, rings, and a cute handbag or crossbody!"
            , "how to look taller": "High-waisted pants, vertical lines, pointy shoes, and monochrome outfits can elongate your frame."
          
            // 🔥 Male-focused
            , "what should a guy wear on a first date": "Dark jeans, a casual button-up, and clean shoes — simple and sharp!"
            , "how should men style overshirts": "Layer over a plain tee with jeans or chinos. Keep colors coordinated and roll up sleeves for a relaxed look."
            , "can men wear jewelry": "Yes! Start with subtle pieces — a watch, ring, or chain. Less is more unless you're going for a bold style."
            , "what’s a good gym outfit for men": "Moisture-wicking shirt, athletic shorts or joggers, and supportive training shoes."
          
            // 🧠 Random & helpful
            , "give me a fashion tip": "Plan your outfits the night before to save time and slay stress-free in the morning. 💡"
            , "what’s trending in fashion right now": "Y2K, oversized fits, chunky sneakers, cargo pants, and co-ords are all over the place!"
            , "is sustainable fashion important": "Totally! It helps reduce waste and supports ethical brands. Small changes make big impacts. 🌱"
            , "can i wear white after labor day": "Yes! Fashion rules were made to be broken. White works all year when styled right. 🤍"
            , "how to build a capsule wardrobe": "Start with basics: neutral tops, jeans, a blazer, and versatile shoes. Add a few standout pieces."
          
            // 💬 Catch-all
            , "default": "I love that question! Want to ask about outfits, colors, events, or something else?"
          };
          
          const getMockResponse = (question: string) => {
            const lowerQ = question.toLowerCase().trim();
          
            for (const key in mockResponses) {
              if (lowerQ.includes(key)) return mockResponses[key];
            }
          
            return mockResponses["default"];
          };
          
      const normalize = (text: string) =>
        text.toLowerCase().replace(/[^\w\s]/gi, '').trim();
  
      const normalizedQuestion = normalize(question);
      const answer = getMockResponse(question);
      
      setResponse(answer);
      setLoading(false);
    }, 1000);
  };
  
  

  return (
    <div className="fashion-assistant-container">
      <h2>💬 Ask the Fashion Assistant</h2>
      <textarea
        placeholder="What's your fashion question?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <button onClick={handleAsk} disabled={loading}>
        {loading ? "Thinking..." : "Ask"}
      </button>
      <div className="response-box">
        {response && <p>{response}</p>}
      </div>
    </div>
  );
};

export default FashionAssistant;
