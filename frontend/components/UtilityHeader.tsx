import Link from "next/link";
import { ArrowLeft, Clapperboard, FileClock, Settings2 } from "lucide-react";

export default function UtilityHeader({ title, subtitle }: { title: string; subtitle: string }) {
    return <header className="border-b border-white/8 bg-[#0d1017]/95 px-5 py-4 md:px-10">
        <div className="mx-auto flex max-w-[1500px] items-center gap-3">
            <Link href="/" className="rounded-lg p-2 text-white/45 hover:bg-white/6 hover:text-white"><ArrowLeft size={19} /></Link>
            <div className="rounded-xl bg-cyan-400 p-2 text-black"><Clapperboard size={22} /></div>
            <div className="min-w-0 flex-1"><h1 className="font-semibold">{title}</h1><p className="text-xs text-white/40">{subtitle}</p></div>
            <Link href="/settings" className="studio-secondary"><Settings2 size={15} /><span className="hidden sm:inline">模型设置</span></Link>
            <Link href="/logs" className="studio-secondary"><FileClock size={15} /><span className="hidden sm:inline">运行日志</span></Link>
        </div>
    </header>;
}
