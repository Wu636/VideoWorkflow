import ProjectWorkspace from "@/components/ProjectWorkspace";

export default async function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
    return <ProjectWorkspace projectId={id} />;
}
