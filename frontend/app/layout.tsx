import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VideoWorkflow Studio",
  description: "从客户需求、分镜设计、MiniMax H3 生成到成片交付的一站式 AI 视频生产系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="dark">
      <body className="font-body antialiased bg-[#050505] text-white selection:bg-cyan-500/30">
        {children}
      </body>
    </html>
  );
}
