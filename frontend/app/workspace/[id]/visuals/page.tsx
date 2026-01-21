import VisualDirector from "@/components/VisualDirector";

interface PageProps {
    params: Promise<{ id: string }>;
}

export default async function VisualsPage({ params }: PageProps) {
    const { id } = await params;
    return (
        <main className="min-h-screen py-12">
            <VisualDirector sessionId={id} />
        </main>
    );
}
