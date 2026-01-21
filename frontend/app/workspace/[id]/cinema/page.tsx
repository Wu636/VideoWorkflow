import Cinema from "@/components/Cinema";

interface PageProps {
    params: Promise<{ id: string }>;
}

export default async function CinemaPage({ params }: PageProps) {
    const { id } = await params;
    return (
        <main className="min-h-screen py-12">
            <Cinema sessionId={id} />
        </main>
    );
}
