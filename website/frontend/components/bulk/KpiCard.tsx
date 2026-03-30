"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface KpiCardProps {
    label: string;
    value: number;
    icon: React.ReactNode;
    sub?: string;
    color?: "emerald" | "red" | "amber" | "orange";
}

const COLOR_MAP: Record<string, string> = {
    emerald: "text-emerald-600 dark:text-emerald-400",
    red: "text-red-600 dark:text-red-400",
    amber: "text-amber-600 dark:text-amber-400",
    orange: "text-orange-600 dark:text-orange-400",
};

export function KpiCard({ label, value, icon, sub, color }: KpiCardProps) {
    const colorClass = color ? COLOR_MAP[color] : "";
    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                    {label}
                </CardTitle>
                {icon}
            </CardHeader>
            <CardContent>
                <div className={`text-2xl font-bold ${colorClass}`}>
                    {value.toLocaleString()}
                </div>
                {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
            </CardContent>
        </Card>
    );
}

export function pct(n: number, counts: Record<string, number>): string {
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (total === 0) return "0%";
    return `${((n / total) * 100).toFixed(1)}% of all aspect predictions`;
}
