import type { BulkPredictResult, SentimentLabel } from "@/lib/types";

/**
 * Parse a raw CSV string into headers and rows.
 * Handles quoted fields with embedded commas.
 */
export function parseCSV(raw: string): { headers: string[]; rows: string[][] } {
    const lines = raw.split(/\r?\n/).filter((l) => l.trim() !== "");
    if (lines.length === 0) return { headers: [], rows: [] };

    function parseLine(line: string): string[] {
        const result: string[] = [];
        let current = "";
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            if (line[i] === '"') {
                inQuotes = !inQuotes;
            } else if (line[i] === "," && !inQuotes) {
                result.push(current.trim());
                current = "";
            } else {
                current += line[i];
            }
        }
        result.push(current.trim());
        return result;
    }

    const headers = parseLine(lines[0]);
    const rows = lines.slice(1).map(parseLine);
    return { headers, rows };
}

/** Download bulk prediction results as a CSV file. */
export function exportResultsCSV(result: BulkPredictResult, aspectNames: string[]) {
    const headerRow = [
        "review_index",
        "text_preview",
        "conflict_prob",
        "is_mixed",
        ...aspectNames.flatMap((a) => [`${a}_label`, `${a}_confidence`]),
    ];

    const dataRows = result.rows.map((row) => {
        const byName = row.aspects.reduce(
            (acc, a) => ({ ...acc, [a.name]: a }),
            {} as Record<string, { label: string; confidence: number }>
        );
        const nonNull = row.aspects.map((a) => a.label).filter((l) => l !== "NULL");
        const isMixed = nonNull.includes("POS") && nonNull.includes("NEG");
        return [
            row.review_index,
            `"${(row.text || "").replace(/"/g, '""').slice(0, 100)}"`,
            row.conflict_prob.toFixed(3),
            isMixed ? "Yes" : "No",
            ...aspectNames.flatMap((a) => [
                byName[a]?.label ?? "NULL",
                (byName[a]?.confidence ?? 0).toFixed(3),
            ]),
        ];
    });

    const csv = [headerRow, ...dataRows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bulk_sentiment_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}
