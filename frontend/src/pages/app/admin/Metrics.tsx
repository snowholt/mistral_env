import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  Server,
  Cpu,
  HardDrive,
  Zap,
  Users,
  MessageSquare,
  Clock,
  TrendingUp,
  TrendingDown,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Mic,
  Bot,
  Database,
  Play,
  Gauge,
} from "lucide-react";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  memory_total_gb: number;
  memory_used_gb: number;
  disk_usage: number;
  disk_total_gb: number;
  disk_used_gb: number;
  gpu_usage: number;
  gpu_memory_usage: number;
  gpu_memory_total_gb: number;
  gpu_memory_used_gb: number;
  uptime_hours: number;
}

interface PlatformMetrics {
  total_users: number;
  active_users_24h: number;
  total_businesses: number;
  total_messages_today: number;
  total_messages_week: number;
  total_voice_sessions: number;
  avg_response_time_ms: number;
  total_kb_documents: number;
  total_kb_chunks: number;
  api_requests_today: number;
  error_rate_percent: number;
}

interface MetricsResponse {
  system: SystemMetrics;
  platform: PlatformMetrics;
  timestamp: string;
}

interface GPUBenchmarkResult {
  tokens_per_second: number;
  total_tokens: number;
  inference_time_seconds: number;
  gpu_name: string;
  gpu_memory_used_gb: number;
  gpu_memory_total_gb: number;
  model_name: string;
  prompt_tokens: number;
  completion_tokens: number;
  runs: number;
}

export default function AdminMetrics() {
  const [timeRange, setTimeRange] = useState("24h");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [benchmarkResult, setBenchmarkResult] = useState<GPUBenchmarkResult | null>(null);

  const { data, isLoading, error, refetch, isFetching } = useQuery<MetricsResponse>({
    queryKey: ["admin-metrics", timeRange],
    queryFn: async () => {
      return api.get<MetricsResponse>(`/api/v1/admin/metrics?range=${timeRange}`);
    },
    refetchInterval: autoRefresh ? 30000 : false, // Refresh every 30 seconds
  });

  // GPU Benchmark mutations
  const quickBenchmarkMutation = useMutation({
    mutationFn: async () => {
      return api.get<GPUBenchmarkResult>('/api/v1/admin/benchmark/gpu/quick');
    },
    onSuccess: (result) => {
      setBenchmarkResult(result);
      toast.success(`Benchmark complete: ${result.tokens_per_second.toFixed(1)} tokens/sec`);
    },
    onError: (error: any) => {
      toast.error(error.detail || 'Benchmark failed');
    },
  });

  const fullBenchmarkMutation = useMutation({
    mutationFn: async () => {
      return api.post<GPUBenchmarkResult>('/api/v1/admin/benchmark/gpu', { runs: 3 });
    },
    onSuccess: (result) => {
      setBenchmarkResult(result);
      toast.success(`Full benchmark complete: ${result.tokens_per_second.toFixed(1)} tokens/sec`);
    },
    onError: (error: any) => {
      toast.error(error.detail || 'Benchmark failed');
    },
  });

  // Force refetch on mount
  useEffect(() => {
    refetch();
  }, []);

  const formatBytes = (gb: number) => {
    return `${gb.toFixed(1)} GB`;
  };

  const getUsageColor = (usage: number) => {
    if (usage >= 90) return "text-red-600";
    if (usage >= 70) return "text-amber-600";
    return "text-green-600";
  };

  const getProgressColor = (usage: number) => {
    if (usage >= 90) return "bg-red-500";
    if (usage >= 70) return "bg-amber-500";
    return "bg-green-500";
  };

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900">Failed to load metrics</h2>
          <p className="text-gray-600 mt-2">Please try again later</p>
          <Button onClick={() => refetch()} className="mt-4">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Platform Metrics</h1>
          <p className="text-gray-600 mt-1">Monitor system health and platform performance</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Auto-refresh:</span>
            <Button
              variant={autoRefresh ? "default" : "outline"}
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              {autoRefresh ? "On" : "Off"}
            </Button>
          </div>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[130px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : (
        <>
          {/* System Health */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                System Health
              </CardTitle>
              <CardDescription>
                Server resource utilization
                {data?.system.uptime_hours && (
                  <Badge variant="outline" className="ml-2">
                    Uptime: {Math.floor(data.system.uptime_hours / 24)}d {Math.floor(data.system.uptime_hours % 24)}h
                  </Badge>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* CPU */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium">CPU</span>
                    </div>
                    <span className={`text-sm font-bold ${getUsageColor(data?.system.cpu_usage || 0)}`}>
                      {data?.system.cpu_usage.toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={data?.system.cpu_usage || 0}
                    className="h-2"
                  />
                </div>

                {/* Memory */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium">Memory</span>
                    </div>
                    <span className={`text-sm font-bold ${getUsageColor(data?.system.memory_usage || 0)}`}>
                      {formatBytes(data?.system.memory_used_gb || 0)} / {formatBytes(data?.system.memory_total_gb || 0)}
                    </span>
                  </div>
                  <Progress
                    value={data?.system.memory_usage || 0}
                    className="h-2"
                  />
                </div>

                {/* Disk */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <HardDrive className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium">Disk</span>
                    </div>
                    <span className={`text-sm font-bold ${getUsageColor(data?.system.disk_usage || 0)}`}>
                      {formatBytes(data?.system.disk_used_gb || 0)} / {formatBytes(data?.system.disk_total_gb || 0)}
                    </span>
                  </div>
                  <Progress
                    value={data?.system.disk_usage || 0}
                    className="h-2"
                  />
                </div>

                {/* GPU */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium">GPU</span>
                    </div>
                    <span className={`text-sm font-bold ${getUsageColor(data?.system.gpu_usage || 0)}`}>
                      {data?.system.gpu_usage.toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={data?.system.gpu_usage || 0}
                    className="h-2"
                  />
                  <p className="text-xs text-gray-500">
                    VRAM: {formatBytes(data?.system.gpu_memory_used_gb || 0)} / {formatBytes(data?.system.gpu_memory_total_gb || 0)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Platform Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Users</p>
                    <p className="text-2xl font-bold">{data?.platform.total_users.toLocaleString()}</p>
                    <p className="text-sm text-green-600 flex items-center gap-1 mt-1">
                      <TrendingUp className="h-3 w-3" />
                      {data?.platform.active_users_24h} active today
                    </p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-lg">
                    <Users className="h-6 w-6 text-blue-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Messages Today</p>
                    <p className="text-2xl font-bold">{data?.platform.total_messages_today.toLocaleString()}</p>
                    <p className="text-sm text-gray-500 mt-1">
                      {data?.platform.total_messages_week.toLocaleString()} this week
                    </p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-lg">
                    <MessageSquare className="h-6 w-6 text-green-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Avg Response Time</p>
                    <p className="text-2xl font-bold">{data?.platform.avg_response_time_ms}ms</p>
                    <p className="text-sm text-gray-500 mt-1">
                      {data?.platform.api_requests_today.toLocaleString()} API calls
                    </p>
                  </div>
                  <div className="p-3 bg-purple-100 rounded-lg">
                    <Clock className="h-6 w-6 text-purple-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Error Rate</p>
                    <p className={`text-2xl font-bold ${data?.platform.error_rate_percent && data.platform.error_rate_percent > 1 ? "text-red-600" : "text-green-600"}`}>
                      {data?.platform.error_rate_percent.toFixed(2)}%
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      {data?.platform.error_rate_percent && data.platform.error_rate_percent > 1 ? (
                        <span className="flex items-center gap-1 text-red-600">
                          <TrendingUp className="h-3 w-3" /> Above threshold
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-green-600">
                          <TrendingDown className="h-3 w-3" /> Healthy
                        </span>
                      )}
                    </p>
                  </div>
                  <div className={`p-3 rounded-lg ${data?.platform.error_rate_percent && data.platform.error_rate_percent > 1 ? "bg-red-100" : "bg-green-100"}`}>
                    <AlertTriangle className={`h-6 w-6 ${data?.platform.error_rate_percent && data.platform.error_rate_percent > 1 ? "text-red-600" : "text-green-600"}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Service Stats */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Mic className="h-5 w-5" />
                  Voice Service
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Active Sessions</span>
                    <span className="font-bold">{data?.platform.total_voice_sessions || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Businesses</span>
                    <span className="font-bold">{data?.platform.total_businesses || 0}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Bot className="h-5 w-5" />
                  AI Inference
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">GPU Utilization</span>
                    <span className={`font-bold ${getUsageColor(data?.system.gpu_usage || 0)}`}>
                      {data?.system.gpu_usage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">VRAM Used</span>
                    <span className="font-bold">
                      {formatBytes(data?.system.gpu_memory_used_gb || 0)}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Database className="h-5 w-5" />
                  Knowledge Base
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Documents</span>
                    <span className="font-bold">{data?.platform.total_kb_documents.toLocaleString() || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Chunks (Vectors)</span>
                    <span className="font-bold">{data?.platform.total_kb_chunks.toLocaleString() || 0}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* GPU Benchmark */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Gauge className="h-5 w-5" />
                    GPU Benchmark
                  </CardTitle>
                  <CardDescription>
                    Measure LLM inference performance (tokens per second)
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => quickBenchmarkMutation.mutate()}
                    disabled={quickBenchmarkMutation.isPending || fullBenchmarkMutation.isPending}
                  >
                    {quickBenchmarkMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Quick Test
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => fullBenchmarkMutation.mutate()}
                    disabled={quickBenchmarkMutation.isPending || fullBenchmarkMutation.isPending}
                  >
                    {fullBenchmarkMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Gauge className="h-4 w-4 mr-2" />
                    )}
                    Full Benchmark
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {benchmarkResult ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div className="text-center">
                    <p className="text-sm text-gray-500">Tokens/Second</p>
                    <p className="text-3xl font-bold text-primary">
                      {benchmarkResult.tokens_per_second.toFixed(1)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-500">Total Tokens</p>
                    <p className="text-2xl font-semibold">
                      {benchmarkResult.total_tokens}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-500">Inference Time</p>
                    <p className="text-2xl font-semibold">
                      {benchmarkResult.inference_time_seconds.toFixed(2)}s
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-500">GPU</p>
                    <p className="text-lg font-medium truncate" title={benchmarkResult.gpu_name}>
                      {benchmarkResult.gpu_name.split(' ').slice(0, 3).join(' ')}
                    </p>
                  </div>
                  <div className="col-span-2 md:col-span-4 pt-4 border-t">
                    <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                      <span>Model: <strong>{benchmarkResult.model_name}</strong></span>
                      <span>Prompt: <strong>{benchmarkResult.prompt_tokens} tokens</strong></span>
                      <span>Completion: <strong>{benchmarkResult.completion_tokens} tokens</strong></span>
                      <span>Runs: <strong>{benchmarkResult.runs}</strong></span>
                      <span>VRAM: <strong>{formatBytes(benchmarkResult.gpu_memory_used_gb)} / {formatBytes(benchmarkResult.gpu_memory_total_gb)}</strong></span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Gauge className="h-12 w-12 mx-auto mb-4 opacity-30" />
                  <p>Run a benchmark to measure GPU inference performance</p>
                  <p className="text-sm mt-1">Quick test: ~5 seconds • Full benchmark: ~30 seconds</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Last Updated */}
          <p className="text-sm text-gray-500 text-center">
            Last updated: {data?.timestamp ? new Date(data.timestamp).toLocaleString() : "N/A"}
          </p>
        </>
      )}
    </div>
  );
}
