import ScriptEditor from "@/components/ScriptEditor";

interface PageProps {
    params: Promise<{ id: string }>;
}

export default async function ScriptPage({ params }: PageProps) {
    const { id } = await params;
    return (
        <main className="min-h-screen py-12">
            <ScriptEditor sessionId={id} />
        </main>
    );
}
