import CreationHub from "@/components/CreationHub";

export default function LegacyHome() {
    return (
        <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-void-black p-4">
            <div className="scanlines" />
            <div className="absolute left-[-10%] top-[-10%] h-[40%] w-[40%] rounded-full bg-neon-cyan/20 blur-[150px]" />
            <div className="absolute bottom-[-10%] right-[-10%] h-[40%] w-[40%] rounded-full bg-neon-purple/20 blur-[150px]" />
            <CreationHub />
        </main>
    );
}
