export interface Scene {
    id: number;
    duration: number;
    narrative: string;
    visual_prompt: string;
    motion_prompt: string;
    image_path?: string;
    video_path?: string;
    image_status?: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
    video_status?: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
    error_message?: string;
}

export interface Storyboard {
    topic: string;
    scenes: Scene[];
}

export interface SessionResponse {
    session_id: string;
    status: string;
    storyboard?: Storyboard;
}

export const VIRAL_TEMPLATES = [
    { id: "viral_reversal", name: "神反转剧", description: "铺垫-误会-反转-结局" },
    { id: "cute_pets", name: "萌宠日常", description: "治愈-搞笑-互动-卖萌" },
    { id: "emotional", name: "情感共鸣", description: "深夜-独白-扎心-治愈" },
    { id: "scifi", name: "科幻脑洞", description: "未来-黑科技-反乌托邦" },
    { id: "funny", name: "搞笑剧场", description: "生活-夸张-玩梗-爆笑" },
];
