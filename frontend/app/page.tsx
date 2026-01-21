import CreationHub from "@/components/CreationHub";

export default function Home() {
  return (
    <main className="min-h-screen relative overflow-hidden flex items-center justify-center p-4 bg-void-black">
      {/* Cyberpunk Background Elements */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-200 contrast-200 mix-blend-overlay pointer-events-none"></div>
      <div className="scanlines"></div>

      {/* Glow Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-neon-cyan/20 rounded-full blur-[150px] animate-pulse-slow" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-neon-purple/20 rounded-full blur-[150px] animate-pulse-slow delay-1000" />

      <CreationHub />
    </main>
  );
}
