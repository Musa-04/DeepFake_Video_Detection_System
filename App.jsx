// import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import DetectVideo from "./pages/DetectVideo";

function Home() {
  const cards = [
    {
      title: "About & Key Features",
      content: (
        <div className="text-gray-300 text-sm md:text-base leading-relaxed space-y-3">
          <p>
            The DeepFake Video Detection System is an AI-powered tool that analyzes uploaded
            videos to determine whether they are{" "}
            <span className="font-semibold text-red-400">REAL</span> or{" "}
            <span className="font-semibold text-red-400">FAKE</span>. 
            It helps users verify content authenticity and avoid misinformation.
          </p>
          <ul className="list-disc list-inside space-y-1 text-left">
            <li>AI-driven frame-by-frame DeepFake analysis</li>
            <li>Accurate classification with confidence score</li>
            <li>Lightweight and easy-to-use interface</li>
          </ul>
        </div>
      ),
    },
    {
      title: "How to Use",
      content: (
        <ul className="text-gray-300 text-sm md:text-base space-y-2 leading-relaxed text-left">
          <li><b>1.</b> Click on <span className="font-semibold text-white ">Start Detection</span>.</li>
          <li><b>2.</b> Upload your video file (.mp4, .avi, .mov).</li>
          <li><b>3.</b> Click <span className="font-semibold text-white">Analyze Video</span>.</li>
          <li><b>4.</b>The AI extracts frames and analyzes them.</li>
          <li><b>5.</b>View results: <span className="font-semibold text-red-400">REAL</span> or{" "}
              <span className="font-semibold text-red-400">FAKE</span> with confidence score <span className="font-semibold text-red-400">Top Suspicious Frames</span></li>
        </ul>
      ),
    },
  ];

  return (
    <section className="flex flex-col items-center justify-center text-center mt-24 px-6 md:mt-28">
      
      {/* Title */}
      <h2 className="text-4xl md:text-5xl font-extrabold mb-10 drop-shadow-[0_0_10px_#00f5d4]">
        DeepFake Video Detection System
      </h2>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl w-full mb-10">
        {cards.map((card, index) => (
          <div
            key={index}
            className="bg-[#0f182c]/70 border border-gray-700 rounded-xl p-6 shadow-[0_0_20px_#00f5d455] hover:shadow-[0_0_25px_#00f5d480] transition-all duration-300 hover:scale-[1.02]"
          >
            <h3 className="text-xl font-semibold mb-4 text-teal-300">
              {card.title}
            </h3>
            {card.content}
          </div>
        ))}
      </div>

      {/* Start Button */}
      <Link
        to="/detect-video"
        className="bg-teal-400 text-black font-semibold mb-3 px-8 py-3 rounded-lg hover:bg-teal-300 transition-all shadow-lg"
      >
        START DETECTION
      </Link>
    </section>
  );
}



function App() {
  return (
    <div className="bg-dark min-h-screen text-white font-poppins flex flex-col">
      {/* Navbar */}
      <nav className="flex justify-between items-center px-6 md:px-10 py-5 border-b border-gray-700 bg-dark/90 sticky top-0 z-50 backdrop-blur-md">
        {/* Logo */}
        <Link
          to="/"
          className="text-xl md:text-2xl font-bold text-neon tracking-wider cursor-pointer hover:drop-shadow-[0_0_10px_#00f5d4] transition-all duration-300"
        >
          DFdetect
        </Link>
      </nav>

      {/* Routes */}
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/detect-video" element={<DetectVideo />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="text-center text-gray-400 py-8 md:py-10 border-t border-gray-700 mt-auto bg-[#0b1221]">
        <div className="max-w-4xl mx-auto px-4">
          <p className="text-sm md:text-base text-gray-400 mb-2">
            © 2025{" "}
            <span className="font-semibold text-white">DFdetect</span>
          </p>

          <p className="text-sm md:text-base font-medium tracking-wide">
            Developed by{" "}
            <a
              href="https://www.linkedin.com/in/mustafeez-shaikh-50b786306"
              target="_blank"
              rel="noopener noreferrer"
              className="text-neon hover:text-teal-400 transition-colors duration-300 font-semibold"
            >
              Mustafeez
            </a>{" "}
            |{" "}
            <a
              href="https://www.linkedin.com/in/sakib-goundi-48497426a" 
              target="_blank"
              rel="noopener noreferrer"
              className="text-neon hover:text-teal-400 transition-colors duration-300 font-semibold"
            >
              Sakib
            </a>{" "}
            |{" "}
            <a
              href="https://www.linkedin.com/in/sanjana-jinaral-359876268" 
              target="_blank"
              rel="noopener noreferrer"
              className="text-neon hover:text-teal-400 transition-colors duration-300 font-semibold"
            >
              Sanjana
            </a>{" "}
            |{" "}
            <a
              href="https://www.linkedin.com/in/sanskriti-singh-825301263?" 
              target="_blank"
              rel="noopener noreferrer"
              className="text-neon hover:text-teal-400 transition-colors duration-300 font-semibold"
            >
              Sanskriti
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;



