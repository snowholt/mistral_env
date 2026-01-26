import { useState, useEffect } from "react";
import { useLanguage } from "@/hooks/useLanguage";
import { adminDemoApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { RefreshCw, Eye, CheckCircle, XCircle, Loader2, Calendar, MessageSquare } from "lucide-react";
import { format } from "date-fns";

interface DemoRequest {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  phone: string | null;
  company: string | null;
  company_size: string | null;
  message: string | null;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  updated_at: string;
  admin_notes: string | null;
  assigned_to_admin_id: number | null;
  scheduled_follow_up: string | null;
  assigned_to?: { id: number; email: string; full_name: string };
}

interface GuestUser {
  id: number;
  email: string;
  is_active: boolean;
  max_conversations: number;
  conversations_used: number;
  expires_at: string;
  created_at: string;
  is_expired: boolean;
  is_limit_reached: boolean;
  can_access: boolean;
  days_remaining: number;
  conversations_remaining: number;
}

export default function DemoRequests() {
  const { language } = useLanguage();
  const [loading, setLoading] = useState(true);
  const [requests, setRequests] = useState<DemoRequest[]>([]);
  const [guestUsers, setGuestUsers] = useState<GuestUser[]>([]);
  const [totalRequests, setTotalRequests] = useState(0);
  const [totalGuests, setTotalGuests] = useState(0);
  const [statusFilter, setStatusFilter] = useState<'pending' | 'approved' | 'rejected' | 'all'>('all');
  const [selectedRequest, setSelectedRequest] = useState<DemoRequest | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [approveModalOpen, setApproveModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'requests' | 'guests'>('requests');

  // Approve form state
  const [maxConversations, setMaxConversations] = useState<number>(10);
  const [daysValid, setDaysValid] = useState<number>(7);
  const [adminNotes, setAdminNotes] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadDemoRequests();
    loadGuestUsers();
  }, [statusFilter]);

  const loadDemoRequests = async () => {
    setLoading(true);
    try {
      const params = statusFilter !== 'all' ? { status: statusFilter } : {};
      const response = await adminDemoApi.listDemoRequests(params);
      setRequests(response.items || []);
      setTotalRequests(response.total || 0);
    } catch (error) {
      console.error("Failed to load demo requests:", error);
      toast.error(language === 'ar' ? 'فشل تحميل طلبات التجربة' : 'Failed to load demo requests');
    } finally {
      setLoading(false);
    }
  };

  const loadGuestUsers = async () => {
    try {
      const response = await adminDemoApi.listGuestUsers();
      setGuestUsers(response.items || []);
      setTotalGuests(response.total || 0);
    } catch (error) {
      console.error("Failed to load guest users:", error);
    }
  };

  const handleViewDetails = (request: DemoRequest) => {
    setSelectedRequest(request);
    setDetailModalOpen(true);
  };

  const handleApproveClick = (request: DemoRequest) => {
    setSelectedRequest(request);
    setAdminNotes("");
    setMaxConversations(10);
    setDaysValid(7);
    setApproveModalOpen(true);
  };

  const handleApprove = async () => {
    if (!selectedRequest) return;

    setSubmitting(true);
    try {
      await adminDemoApi.approveDemoRequest(selectedRequest.id, {
        max_conversations: maxConversations,
        days_valid: daysValid,
        admin_notes: adminNotes || undefined
      });

      toast.success(
        language === 'ar' 
          ? 'تم قبول الطلب وإرسال رسالة الوصول' 
          : 'Request approved and access email sent'
      );

      setApproveModalOpen(false);
      setSelectedRequest(null);
      loadDemoRequests();
      loadGuestUsers();
    } catch (error: any) {
      console.error("Failed to approve request:", error);
      toast.error(
        error?.response?.data?.detail || 
        (language === 'ar' ? 'فشل قبول الطلب' : 'Failed to approve request')
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleReject = async (request: DemoRequest) => {
    const notes = prompt(language === 'ar' ? 'ملاحظات الرفض (اختياري):' : 'Rejection notes (optional):');
    
    try {
      await adminDemoApi.rejectDemoRequest(request.id, notes || undefined);
      toast.success(language === 'ar' ? 'تم رفض الطلب' : 'Request rejected');
      loadDemoRequests();
    } catch (error) {
      console.error("Failed to reject request:", error);
      toast.error(language === 'ar' ? 'فشل رفض الطلب' : 'Failed to reject request');
    }
  };

  const handleDeleteRequest = async (id: number) => {
    if (!confirm(language === 'ar' ? 'هل أنت متأكد من حذف هذا الطلب؟' : 'Are you sure you want to delete this request?')) {
      return;
    }

    try {
      await adminDemoApi.deleteDemoRequest(id);
      toast.success(language === 'ar' ? 'تم حذف الطلب' : 'Request deleted');
      loadDemoRequests();
    } catch (error) {
      console.error("Failed to delete request:", error);
      toast.error(language === 'ar' ? 'فشل حذف الطلب' : 'Failed to delete request');
    }
  };

  const handleToggleGuestActive = async (guest: GuestUser) => {
    try {
      await adminDemoApi.updateGuestUser(guest.id, { is_active: !guest.is_active });
      toast.success(
        language === 'ar' 
          ? (guest.is_active ? 'تم تعطيل الضيف' : 'تم تفعيل الضيف')
          : (guest.is_active ? 'Guest deactivated' : 'Guest activated')
      );
      loadGuestUsers();
    } catch (error) {
      console.error("Failed to update guest:", error);
      toast.error(language === 'ar' ? 'فشل تحديث الضيف' : 'Failed to update guest');
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline", label: string }> = {
      pending: { variant: "secondary", label: language === 'ar' ? 'قيد الانتظار' : 'Pending' },
      approved: { variant: "default", label: language === 'ar' ? 'مقبول' : 'Approved' },
      rejected: { variant: "destructive", label: language === 'ar' ? 'مرفوض' : 'Rejected' }
    };

    const config = variants[status] || variants.pending;
    return <Badge variant={config.variant}>{config.label}</Badge>;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">
            {language === 'ar' ? 'طلبات التجربة' : 'Demo Requests'}
          </h1>
          <p className="text-muted-foreground mt-1">
            {language === 'ar' 
              ? 'إدارة طلبات التجربة والمستخدمين الضيوف' 
              : 'Manage demo requests and guest users'
            }
          </p>
        </div>
        <Button onClick={() => { loadDemoRequests(); loadGuestUsers(); }} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          {language === 'ar' ? 'تحديث' : 'Refresh'}
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{language === 'ar' ? 'إجمالي الطلبات' : 'Total Requests'}</CardDescription>
            <CardTitle className="text-3xl">{totalRequests}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{language === 'ar' ? 'قيد الانتظار' : 'Pending'}</CardDescription>
            <CardTitle className="text-3xl text-yellow-600">
              {requests.filter(r => r.status === 'pending').length}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{language === 'ar' ? 'مقبول' : 'Approved'}</CardDescription>
            <CardTitle className="text-3xl text-green-600">
              {requests.filter(r => r.status === 'approved').length}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{language === 'ar' ? 'مستخدمين ضيوف' : 'Guest Users'}</CardDescription>
            <CardTitle className="text-3xl text-blue-600">{totalGuests}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'requests' | 'guests')}>
        <TabsList>
          <TabsTrigger value="requests">
            {language === 'ar' ? 'الطلبات' : 'Requests'} ({totalRequests})
          </TabsTrigger>
          <TabsTrigger value="guests">
            {language === 'ar' ? 'الضيوف' : 'Guests'} ({totalGuests})
          </TabsTrigger>
        </TabsList>

        {/* Demo Requests Tab */}
        <TabsContent value="requests">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>{language === 'ar' ? 'طلبات التجربة' : 'Demo Requests'}</CardTitle>
                <Select
                  value={statusFilter}
                  onValueChange={(value: any) => setStatusFilter(value)}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">{language === 'ar' ? 'الكل' : 'All'}</SelectItem>
                    <SelectItem value="pending">{language === 'ar' ? 'قيد الانتظار' : 'Pending'}</SelectItem>
                    <SelectItem value="approved">{language === 'ar' ? 'مقبول' : 'Approved'}</SelectItem>
                    <SelectItem value="rejected">{language === 'ar' ? 'مرفوض' : 'Rejected'}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : requests.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  {language === 'ar' ? 'لا توجد طلبات' : 'No requests found'}
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{language === 'ar' ? 'الاسم' : 'Name'}</TableHead>
                      <TableHead>{language === 'ar' ? 'البريد الإلكتروني' : 'Email'}</TableHead>
                      <TableHead>{language === 'ar' ? 'الشركة' : 'Company'}</TableHead>
                      <TableHead>{language === 'ar' ? 'الحالة' : 'Status'}</TableHead>
                      <TableHead>{language === 'ar' ? 'التاريخ' : 'Date'}</TableHead>
                      <TableHead className="text-right">{language === 'ar' ? 'الإجراءات' : 'Actions'}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {requests.map((request) => (
                      <TableRow key={request.id}>
                        <TableCell className="font-medium">
                          {request.first_name} {request.last_name}
                        </TableCell>
                        <TableCell>{request.email}</TableCell>
                        <TableCell>{request.company || '-'}</TableCell>
                        <TableCell>{getStatusBadge(request.status)}</TableCell>
                        <TableCell>
                          {format(new Date(request.created_at), 'MMM dd, yyyy')}
                        </TableCell>
                        <TableCell className="text-right space-x-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleViewDetails(request)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          {request.status === 'pending' && (
                            <>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="text-green-600 hover:text-green-700"
                                onClick={() => handleApproveClick(request)}
                              >
                                <CheckCircle className="h-4 w-4" />
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="text-red-600 hover:text-red-700"
                                onClick={() => handleReject(request)}
                              >
                                <XCircle className="h-4 w-4" />
                              </Button>
                            </>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Guest Users Tab */}
        <TabsContent value="guests">
          <Card>
            <CardHeader>
              <CardTitle>{language === 'ar' ? 'المستخدمين الضيوف' : 'Guest Users'}</CardTitle>
            </CardHeader>
            <CardContent>
              {guestUsers.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  {language === 'ar' ? 'لا يوجد مستخدمين ضيوف' : 'No guest users'}
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{language === 'ar' ? 'البريد الإلكتروني' : 'Email'}</TableHead>
                      <TableHead>{language === 'ar' ? 'الحالة' : 'Status'}</TableHead>
                      <TableHead>{language === 'ar' ? 'الاستخدام' : 'Usage'}</TableHead>
                      <TableHead>{language === 'ar' ? 'ينتهي في' : 'Expires'}</TableHead>
                      <TableHead className="text-right">{language === 'ar' ? 'الإجراءات' : 'Actions'}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {guestUsers.map((guest) => (
                      <TableRow key={guest.id}>
                        <TableCell className="font-medium">{guest.email}</TableCell>
                        <TableCell>
                          <Badge variant={guest.can_access ? "default" : "secondary"}>
                            {guest.can_access 
                              ? (language === 'ar' ? 'نشط' : 'Active')
                              : guest.is_expired 
                                ? (language === 'ar' ? 'منتهي' : 'Expired')
                                : (language === 'ar' ? 'محظور' : 'Inactive')
                            }
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {guest.conversations_used}/{guest.max_conversations}
                        </TableCell>
                        <TableCell>
                          {guest.is_expired 
                            ? (language === 'ar' ? 'منتهي' : 'Expired')
                            : `${guest.days_remaining} ${language === 'ar' ? 'يوم' : 'days'}`
                          }
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleToggleGuestActive(guest)}
                          >
                            {guest.is_active 
                              ? (language === 'ar' ? 'تعطيل' : 'Deactivate')
                              : (language === 'ar' ? 'تفعيل' : 'Activate')
                            }
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Detail Modal */}
      <Dialog open={detailModalOpen} onOpenChange={setDetailModalOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {language === 'ar' ? 'تفاصيل الطلب' : 'Request Details'}
            </DialogTitle>
          </DialogHeader>
          {selectedRequest && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'الاسم الأول' : 'First Name'}</Label>
                  <p className="font-medium">{selectedRequest.first_name}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'اسم العائلة' : 'Last Name'}</Label>
                  <p className="font-medium">{selectedRequest.last_name}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'البريد الإلكتروني' : 'Email'}</Label>
                  <p className="font-medium">{selectedRequest.email}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'الهاتف' : 'Phone'}</Label>
                  <p className="font-medium">{selectedRequest.phone || '-'}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'الشركة' : 'Company'}</Label>
                  <p className="font-medium">{selectedRequest.company || '-'}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'حجم الشركة' : 'Company Size'}</Label>
                  <p className="font-medium">{selectedRequest.company_size || '-'}</p>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'الحالة' : 'Status'}</Label>
                  <div className="mt-1">{getStatusBadge(selectedRequest.status)}</div>
                </div>
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'التاريخ' : 'Date'}</Label>
                  <p className="font-medium">
                    {format(new Date(selectedRequest.created_at), 'PPP')}
                  </p>
                </div>
              </div>
              {selectedRequest.message && (
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'الرسالة' : 'Message'}</Label>
                  <p className="mt-1 text-sm border rounded p-3 bg-muted/50">
                    {selectedRequest.message}
                  </p>
                </div>
              )}
              {selectedRequest.admin_notes && (
                <div>
                  <Label className="text-muted-foreground">{language === 'ar' ? 'ملاحظات المسؤول' : 'Admin Notes'}</Label>
                  <p className="mt-1 text-sm border rounded p-3 bg-muted/50">
                    {selectedRequest.admin_notes}
                  </p>
                </div>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDetailModalOpen(false)}>
              {language === 'ar' ? 'إغلاق' : 'Close'}
            </Button>
            {selectedRequest?.status === 'pending' && (
              <>
                <Button
                  variant="destructive"
                  onClick={() => {
                    setDetailModalOpen(false);
                    handleReject(selectedRequest);
                  }}
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  {language === 'ar' ? 'رفض' : 'Reject'}
                </Button>
                <Button
                  onClick={() => {
                    setDetailModalOpen(false);
                    handleApproveClick(selectedRequest);
                  }}
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  {language === 'ar' ? 'قبول' : 'Approve'}
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Approve Modal */}
      <Dialog open={approveModalOpen} onOpenChange={setApproveModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {language === 'ar' ? 'قبول طلب التجربة' : 'Approve Demo Request'}
            </DialogTitle>
            <DialogDescription>
              {language === 'ar' 
                ? 'قم بتعيين حدود الاستخدام وفترة الصلاحية للمستخدم الضيف'
                : 'Set usage limits and expiration for the guest user'
              }
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="maxConversations">
                {language === 'ar' ? 'عدد المحادثات الأقصى' : 'Max Conversations'}
              </Label>
              <Input
                id="maxConversations"
                type="number"
                min="1"
                value={maxConversations}
                onChange={(e) => setMaxConversations(parseInt(e.target.value))}
              />
              <p className="text-xs text-muted-foreground mt-1">
                {language === 'ar' 
                  ? 'عدد المحادثات المسموح بها قبل انتهاء الصلاحية'
                  : 'Number of conversations allowed before expiry'
                }
              </p>
            </div>
            <div>
              <Label htmlFor="daysValid">
                {language === 'ar' ? 'صالح لمدة (أيام)' : 'Valid For (days)'}
              </Label>
              <Input
                id="daysValid"
                type="number"
                min="1"
                value={daysValid}
                onChange={(e) => setDaysValid(parseInt(e.target.value))}
              />
              <p className="text-xs text-muted-foreground mt-1">
                {language === 'ar' 
                  ? 'عدد الأيام قبل انتهاء صلاحية الوصول'
                  : 'Number of days before access expires'
                }
              </p>
            </div>
            <div>
              <Label htmlFor="adminNotes">
                {language === 'ar' ? 'ملاحظات المسؤول (اختياري)' : 'Admin Notes (optional)'}
              </Label>
              <Textarea
                id="adminNotes"
                value={adminNotes}
                onChange={(e) => setAdminNotes(e.target.value)}
                placeholder={language === 'ar' ? 'ملاحظات داخلية...' : 'Internal notes...'}
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setApproveModalOpen(false)}
              disabled={submitting}
            >
              {language === 'ar' ? 'إلغاء' : 'Cancel'}
            </Button>
            <Button onClick={handleApprove} disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  {language === 'ar' ? 'جاري القبول...' : 'Approving...'}
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  {language === 'ar' ? 'قبول وإرسال البريد' : 'Approve & Send Email'}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
