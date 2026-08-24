import PublicReview from "@/components/PublicReview";

export default async function ReviewPage({ params }: { params: Promise<{ token: string }> }) {
    const { token } = await params;
    return <PublicReview token={token} />;
}
