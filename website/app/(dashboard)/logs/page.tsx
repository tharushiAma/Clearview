"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getLogs, exportLogsAsCSV, exportLogsAsJSON } from "@/lib/api";
import type { LogEntry } from "@/lib/types";
import { Download, FileJson, FileSpreadsheet, RefreshCw } from "lucide-react";

export default function LogsPage() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const data = await getLogs();
      setLogs(data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handleExportCSV = () => {
    exportLogsAsCSV(logs);
  };

  const handleExportJSON = () => {
    exportLogsAsJSON(logs);
  };

  const formatDate = (iso: string) => {
    const date = new Date(iso);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getOutcomeBadgeVariant = (outcome: string) => {
    if (outcome.includes("Positive")) return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
    if (outcome.includes("Negative")) return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400";
    if (outcome.includes("Mixed")) return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400";
    return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Logs</h1>
          <p className="text-muted-foreground">
            History of prediction runs and results
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={fetchLogs} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={handleExportCSV} disabled={logs.length === 0}>
            <FileSpreadsheet className="h-4 w-4 mr-2" />
            Export CSV
          </Button>
          <Button variant="outline" size="sm" onClick={handleExportJSON} disabled={logs.length === 0}>
            <FileJson className="h-4 w-4 mr-2" />
            Export JSON
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            Recent Runs
            {logs.length > 0 && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                ({logs.length} entries)
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <LogsSkeleton />
          ) : logs.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Download className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No logs available</p>
              <p className="text-sm">Run some predictions to see them here</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Text Preview</TableHead>
                    <TableHead>MSR</TableHead>
                    <TableHead>Conflict Prob</TableHead>
                    <TableHead>Outcome</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {logs.map((log) => (
                    <TableRow key={log.id}>
                      <TableCell className="whitespace-nowrap text-muted-foreground">
                        {formatDate(log.timestamp)}
                      </TableCell>
                      <TableCell className="max-w-xs truncate" title={log.textPreview}>
                        {log.textPreview}
                      </TableCell>
                      <TableCell>
                        {log.msrEnabled ? (
                          <Badge variant="secondary" className="text-xs">
                            Enabled
                          </Badge>
                        ) : (
                          <span className="text-muted-foreground text-sm">Off</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <span
                          className={`font-medium ${
                            log.conflictProbability > 0.4
                              ? "text-amber-600 dark:text-amber-400"
                              : "text-muted-foreground"
                          }`}
                        >
                          {(log.conflictProbability * 100).toFixed(1)}%
                        </span>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="secondary"
                          className={getOutcomeBadgeVariant(log.overallOutcome)}
                        >
                          {log.overallOutcome}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function LogsSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex items-center gap-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 flex-1" />
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-12" />
          <Skeleton className="h-6 w-24" />
        </div>
      ))}
    </div>
  );
}
