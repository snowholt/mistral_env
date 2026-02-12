import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  KeyRound,
  Search,
  MoreHorizontal,
  Eye,
  Trash2,
  ShieldCheck,
  ShieldOff,
  Plus,
  Building2,
  History,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Copy,
} from "lucide-react";
import { format, formatDistanceToNow } from "date-fns";
import { toast } from "sonner";

interface Credential {
  id: number;
  customer_id: number;
  customer_name: string | null;
  credential_type: string;
  token_prefix: string;
  scopes: string[] | null;
  is_active: boolean;
  is_revoked: boolean;
  expires_at: string | null;
  last_used_at: string | null;
  use_count: number;
  created_at: string;
}

interface CredentialsResponse {
  credentials: Credential[];
  total: number;
  page: number;
  per_page: number;
}

interface CredentialStats {
  total_credentials: number;
  active_credentials: number;
  revoked_credentials: number;
  expired_credentials: number;
  credentials_by_type: Record<string, number>;
  recent_activity: number;
}

interface AuditLogEntry {
  id: number;
  action: string;
  resource_type: string;
  resource_id: string;
  customer_id: number | null;
  user_id: number | null;
  ip_address: string | null;
  user_agent: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

interface AuditLogResponse {
  logs: AuditLogEntry[];
  total: number;
}

export default function AdminCredentials() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  
  // Dialog states
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [auditDialogOpen, setAuditDialogOpen] = useState(false);
  const [revokeDialogOpen, setRevokeDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<Credential | null>(null);
  
  // Create form state
  const [createForm, setCreateForm] = useState({
    customer_id: "",
    token: "",
    credential_type: "system_user_token",
    scopes: "",
  });

  // Fetch credentials
  const { data, isLoading, error } = useQuery<CredentialsResponse>({
    queryKey: ["admin-credentials", page, search, typeFilter, statusFilter],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: "20",
      });
      if (search) params.append("search", search);
      if (typeFilter !== "all") params.append("credential_type", typeFilter);
      if (statusFilter === "active") params.append("is_active", "true");
      if (statusFilter === "revoked") params.append("is_active", "false");

      return api.get<CredentialsResponse>(`/api/v1/admin/credentials?${params}`);
    },
  });

  // Fetch stats
  const { data: stats } = useQuery<CredentialStats>({
    queryKey: ["admin-credentials-stats"],
    queryFn: () => api.get<CredentialStats>("/api/v1/admin/credentials/stats/summary"),
  });

  // Fetch audit log for selected credential
  const { data: auditData, isLoading: auditLoading } = useQuery<AuditLogResponse>({
    queryKey: ["admin-credential-audit", selectedCredential?.id],
    queryFn: () => api.get<AuditLogResponse>(`/api/v1/admin/credentials/${selectedCredential?.id}/audit-log`),
    enabled: !!selectedCredential && auditDialogOpen,
  });

  // Create credential mutation
  const createMutation = useMutation({
    mutationFn: async (data: typeof createForm) => {
      return api.post("/api/v1/admin/credentials", {
        customer_id: parseInt(data.customer_id),
        token: data.token,
        credential_type: data.credential_type,
        scopes: data.scopes ? data.scopes.split(",").map(s => s.trim()) : null,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-credentials"] });
      queryClient.invalidateQueries({ queryKey: ["admin-credentials-stats"] });
      setCreateDialogOpen(false);
      setCreateForm({ customer_id: "", token: "", credential_type: "system_user_token", scopes: "" });
      toast.success("Credential created successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to create credential: ${error.message}`);
    },
  });

  // Validate credential mutation
  const validateMutation = useMutation({
    mutationFn: async (credentialId: number) => {
      return api.post<{ valid: boolean; error?: string }>(`/api/v1/admin/credentials/${credentialId}/validate`);
    },
    onSuccess: (data) => {
      if (data.valid) {
        toast.success("Token is valid and working");
      } else {
        toast.error(`Token validation failed: ${data.error || "Unknown error"}`);
      }
    },
    onError: (error: Error) => {
      toast.error(`Validation failed: ${error.message}`);
    },
  });

  // Revoke credential mutation
  const revokeMutation = useMutation({
    mutationFn: async ({ credentialId, reason }: { credentialId: number; reason?: string }) => {
      return api.patch(`/api/v1/admin/credentials/${credentialId}/revoke`, { reason });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-credentials"] });
      queryClient.invalidateQueries({ queryKey: ["admin-credentials-stats"] });
      setRevokeDialogOpen(false);
      setSelectedCredential(null);
      toast.success("Credential revoked successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to revoke credential: ${error.message}`);
    },
  });

  // Delete credential mutation
  const deleteMutation = useMutation({
    mutationFn: async (credentialId: number) => {
      return api.delete(`/api/v1/admin/credentials/${credentialId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-credentials"] });
      queryClient.invalidateQueries({ queryKey: ["admin-credentials-stats"] });
      setDeleteDialogOpen(false);
      setSelectedCredential(null);
      toast.success("Credential deleted permanently");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete credential: ${error.message}`);
    },
  });

  const getStatusBadge = (credential: Credential) => {
    if (credential.is_revoked) {
      return <Badge variant="destructive"><ShieldOff className="h-3 w-3 mr-1" /> Revoked</Badge>;
    }
    if (credential.expires_at && new Date(credential.expires_at) < new Date()) {
      return <Badge variant="outline" className="text-orange-600 border-orange-300"><Clock className="h-3 w-3 mr-1" /> Expired</Badge>;
    }
    return <Badge variant="default" className="bg-green-600"><ShieldCheck className="h-3 w-3 mr-1" /> Active</Badge>;
  };

  const getTypeBadge = (type: string) => {
    const colors: Record<string, string> = {
      system_user_token: "bg-purple-100 text-purple-800",
      user_token: "bg-blue-100 text-blue-800",
      page_token: "bg-amber-100 text-amber-800",
      app_token: "bg-green-100 text-green-800",
    };
    const labels: Record<string, string> = {
      system_user_token: "System User",
      user_token: "User Token",
      page_token: "Page Token",
      app_token: "App Token",
    };
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[type] || "bg-gray-100 text-gray-800"}`}>
        {labels[type] || type}
      </span>
    );
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900">Failed to load credentials</h2>
          <p className="text-gray-600 mt-2">Please try again later</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Meta Credentials</h1>
          <p className="text-gray-600 mt-1">Manage encrypted Meta API tokens for all customers</p>
        </div>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Add Credential
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <KeyRound className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Credentials</p>
                <p className="text-2xl font-bold">{stats?.total_credentials || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Active</p>
                <p className="text-2xl font-bold">{stats?.active_credentials || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-red-100 rounded-lg">
                <XCircle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Revoked</p>
                <p className="text-2xl font-bold">{stats?.revoked_credentials || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-100 rounded-lg">
                <Clock className="h-5 w-5 text-amber-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Expired</p>
                <p className="text-2xl font-bold">{stats?.expired_credentials || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters & Table */}
      <Card>
        <CardHeader>
          <CardTitle>Credential List</CardTitle>
          <CardDescription>View and manage all Meta API credentials</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search by customer name or token prefix..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="system_user_token">System User</SelectItem>
                <SelectItem value="user_token">User Token</SelectItem>
                <SelectItem value="page_token">Page Token</SelectItem>
                <SelectItem value="app_token">App Token</SelectItem>
              </SelectContent>
            </Select>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="revoked">Revoked</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Table */}
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Customer</TableHead>
                      <TableHead>Token</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Usage</TableHead>
                      <TableHead>Last Used</TableHead>
                      <TableHead className="w-[50px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data?.credentials?.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={7} className="text-center py-10 text-gray-500">
                          No credentials found
                        </TableCell>
                      </TableRow>
                    ) : (
                      data?.credentials?.map((credential) => (
                        <TableRow key={credential.id}>
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <div className="h-10 w-10 rounded-full bg-gray-100 flex items-center justify-center">
                                <Building2 className="h-5 w-5 text-gray-600" />
                              </div>
                              <div>
                                <p className="font-medium">{credential.customer_name || "Unknown"}</p>
                                <p className="text-sm text-gray-500">ID: {credential.customer_id}</p>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono">
                                {credential.token_prefix}
                              </code>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6"
                                onClick={() => copyToClipboard(credential.token_prefix)}
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                            </div>
                          </TableCell>
                          <TableCell>{getTypeBadge(credential.credential_type)}</TableCell>
                          <TableCell>{getStatusBadge(credential)}</TableCell>
                          <TableCell>
                            <p className="font-medium">{credential.use_count.toLocaleString()}</p>
                            <p className="text-sm text-gray-500">API calls</p>
                          </TableCell>
                          <TableCell>
                            {credential.last_used_at ? (
                              <span title={format(new Date(credential.last_used_at), "PPpp")}>
                                {formatDistanceToNow(new Date(credential.last_used_at), { addSuffix: true })}
                              </span>
                            ) : (
                              <span className="text-gray-400">Never</span>
                            )}
                          </TableCell>
                          <TableCell>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => {
                                  setSelectedCredential(credential);
                                  setDetailsDialogOpen(true);
                                }}>
                                  <Eye className="h-4 w-4 mr-2" />
                                  View Details
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => validateMutation.mutate(credential.id)}
                                  disabled={validateMutation.isPending}
                                >
                                  <RefreshCw className={`h-4 w-4 mr-2 ${validateMutation.isPending ? 'animate-spin' : ''}`} />
                                  Validate Token
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={() => {
                                  setSelectedCredential(credential);
                                  setAuditDialogOpen(true);
                                }}>
                                  <History className="h-4 w-4 mr-2" />
                                  Audit Log
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                {!credential.is_revoked && (
                                  <DropdownMenuItem
                                    className="text-orange-600"
                                    onClick={() => {
                                      setSelectedCredential(credential);
                                      setRevokeDialogOpen(true);
                                    }}
                                  >
                                    <ShieldOff className="h-4 w-4 mr-2" />
                                    Revoke
                                  </DropdownMenuItem>
                                )}
                                <DropdownMenuItem
                                  className="text-red-600"
                                  onClick={() => {
                                    setSelectedCredential(credential);
                                    setDeleteDialogOpen(true);
                                  }}
                                >
                                  <Trash2 className="h-4 w-4 mr-2" />
                                  Delete Permanently
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              {data && data.total > 20 && (
                <div className="flex items-center justify-between mt-4">
                  <p className="text-sm text-gray-600">
                    Showing {(page - 1) * 20 + 1} to {Math.min(page * 20, data.total)} of {data.total}
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(page - 1)}
                      disabled={page === 1}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage(page + 1)}
                      disabled={page * 20 >= data.total}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Create Credential Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Add New Credential</DialogTitle>
            <DialogDescription>
              Create a new encrypted Meta API credential for a customer.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="customer_id">Customer ID</Label>
              <Input
                id="customer_id"
                type="number"
                placeholder="Enter customer ID"
                value={createForm.customer_id}
                onChange={(e) => setCreateForm({ ...createForm, customer_id: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="credential_type">Credential Type</Label>
              <Select
                value={createForm.credential_type}
                onValueChange={(value) => setCreateForm({ ...createForm, credential_type: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="system_user_token">System User Token (Recommended)</SelectItem>
                  <SelectItem value="user_token">User Token</SelectItem>
                  <SelectItem value="page_token">Page Token</SelectItem>
                  <SelectItem value="app_token">App Token</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-gray-500 mt-1">
                System User tokens are permanent and recommended for production use.
              </p>
            </div>
            <div>
              <Label htmlFor="token">Access Token</Label>
              <Textarea
                id="token"
                placeholder="Paste the Meta API token here..."
                value={createForm.token}
                onChange={(e) => setCreateForm({ ...createForm, token: e.target.value })}
                className="font-mono text-sm"
                rows={3}
              />
              <p className="text-xs text-gray-500 mt-1">
                The token will be encrypted before storage.
              </p>
            </div>
            <div>
              <Label htmlFor="scopes">Scopes (Optional)</Label>
              <Input
                id="scopes"
                placeholder="whatsapp_business_management, whatsapp_business_messaging"
                value={createForm.scopes}
                onChange={(e) => setCreateForm({ ...createForm, scopes: e.target.value })}
              />
              <p className="text-xs text-gray-500 mt-1">
                Comma-separated list of OAuth scopes granted to this token.
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMutation.mutate(createForm)}
              disabled={createMutation.isPending || !createForm.customer_id || !createForm.token}
            >
              {createMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Create Credential
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Credential Details Dialog */}
      <Dialog open={detailsDialogOpen} onOpenChange={setDetailsDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Credential Details</DialogTitle>
          </DialogHeader>
          {selectedCredential && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-500">Customer</label>
                  <p className="text-lg">{selectedCredential.customer_name || "Unknown"}</p>
                  <p className="text-sm text-gray-500">ID: {selectedCredential.customer_id}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Type</label>
                  <p className="mt-1">{getTypeBadge(selectedCredential.credential_type)}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Status</label>
                  <p className="mt-1">{getStatusBadge(selectedCredential)}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Token Prefix</label>
                  <code className="block mt-1 bg-gray-100 px-2 py-1 rounded text-sm font-mono">
                    {selectedCredential.token_prefix}
                  </code>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">API Calls</label>
                  <p className="text-lg">{selectedCredential.use_count.toLocaleString()}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Last Used</label>
                  <p className="text-lg">
                    {selectedCredential.last_used_at
                      ? formatDistanceToNow(new Date(selectedCredential.last_used_at), { addSuffix: true })
                      : "Never"}
                  </p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Created</label>
                  <p className="text-lg">{format(new Date(selectedCredential.created_at), "PPP")}</p>
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-500">Expires</label>
                  <p className="text-lg">
                    {selectedCredential.expires_at
                      ? format(new Date(selectedCredential.expires_at), "PPP")
                      : "Never (Permanent)"}
                  </p>
                </div>
              </div>
              {selectedCredential.scopes && selectedCredential.scopes.length > 0 && (
                <div>
                  <label className="text-sm font-medium text-gray-500">Scopes</label>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {selectedCredential.scopes.map((scope: string) => (
                      <Badge key={scope} variant="outline" className="text-xs">
                        {scope}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Audit Log Dialog */}
      <Dialog open={auditDialogOpen} onOpenChange={setAuditDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Audit Log</DialogTitle>
            <DialogDescription>
              History of actions for credential {selectedCredential?.token_prefix}
            </DialogDescription>
          </DialogHeader>
          {auditLoading ? (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : auditData?.logs?.length === 0 ? (
            <p className="text-center text-gray-500 py-8">No audit events found</p>
          ) : (
            <div className="space-y-3">
              {auditData?.logs?.map((log) => (
                <div key={log.id} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline">{log.action}</Badge>
                    <span className="text-sm text-gray-500">
                      {format(new Date(log.created_at), "PPpp")}
                    </span>
                  </div>
                  {log.ip_address && (
                    <p className="text-sm text-gray-500 mt-1">IP: {log.ip_address}</p>
                  )}
                  {log.metadata && Object.keys(log.metadata).length > 0 && (
                    <pre className="text-xs bg-gray-50 p-2 rounded mt-2 overflow-x-auto">
                      {JSON.stringify(log.metadata, null, 2)}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Revoke Confirmation Dialog */}
      <AlertDialog open={revokeDialogOpen} onOpenChange={setRevokeDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Revoke Credential</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to revoke this credential? The token will no longer be usable
              for API calls. This action cannot be undone.
              <br /><br />
              <strong>Token:</strong> {selectedCredential?.token_prefix}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-orange-600 hover:bg-orange-700"
              onClick={() => selectedCredential && revokeMutation.mutate({ credentialId: selectedCredential.id })}
              disabled={revokeMutation.isPending}
            >
              {revokeMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Revoke
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Credential Permanently</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to permanently delete this credential? This action cannot be
              undone and all audit logs will be lost.
              <br /><br />
              <strong>Token:</strong> {selectedCredential?.token_prefix}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => selectedCredential && deleteMutation.mutate(selectedCredential.id)}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete Permanently
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
