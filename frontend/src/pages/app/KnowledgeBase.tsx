import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'sonner';
import {
  Loader2,
  Upload,
  FileText,
  Trash2,
  Search,
  Plus,
  FolderOpen,
  File,
  FileIcon,
  MoreHorizontal,
  RefreshCw,
  BookOpen,
  Database,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';

import { api } from '@/lib/api';
import { useCustomers } from '@/hooks/useAgent';

// Types
interface KnowledgeBase {
  id: number;
  name: string;
  description: string | null;
  document_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string | null;
}

interface Document {
  id: number;
  title: string;
  file_name: string;
  file_type: string;
  file_size: number;
  status: 'pending' | 'processing' | 'indexed' | 'failed';
  chunk_count: number | null;
  error_message: string | null;
  created_at: string;
  processed_at: string | null;
}

interface SearchResult {
  chunk_id: number;
  document_id: number;
  document_title: string;
  content: string;
  score: number;
}

// Helper functions
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileIcon(fileType: string) {
  if (fileType.includes('pdf')) return <FileText className="h-5 w-5 text-red-500" />;
  if (fileType.includes('word') || fileType.includes('docx')) return <FileText className="h-5 w-5 text-blue-500" />;
  if (fileType.includes('text') || fileType.includes('markdown')) return <FileIcon className="h-5 w-5 text-gray-500" />;
  return <File className="h-5 w-5" />;
}

function getStatusBadge(status: Document['status']) {
  switch (status) {
    case 'pending':
      return <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">Pending</Badge>;
    case 'processing':
      return <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">Processing</Badge>;
    case 'indexed':
      return <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">Indexed</Badge>;
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

export default function KnowledgeBase() {
  // State
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKB, setSelectedKB] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newKBName, setNewKBName] = useState('');
  const [newKBDescription, setNewKBDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const { data: customers } = useCustomers();
  const customerId = customers?.[0]?.id;

  // Fetch knowledge bases
  const fetchKnowledgeBases = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await api.get<KnowledgeBase[]>('/api/v1/kb');
      setKnowledgeBases(data);
      if (data.length > 0 && !selectedKB) {
        setSelectedKB(data[0]);
      }
    } catch (error: any) {
      toast.error(error.detail || 'Failed to load knowledge bases');
    } finally {
      setIsLoading(false);
    }
  }, [selectedKB]);

  // Fetch documents for selected KB
  const fetchDocuments = useCallback(async () => {
    if (!selectedKB) return;
    try {
      const data = await api.get<Document[]>(`/api/v1/kb/${selectedKB.id}/documents`);
      setDocuments(data);
    } catch (error: any) {
      toast.error(error.detail || 'Failed to load documents');
    }
  }, [selectedKB]);

  // Effects
  useEffect(() => {
    fetchKnowledgeBases();
  }, []);

  useEffect(() => {
    if (selectedKB) {
      fetchDocuments();
    }
  }, [selectedKB, fetchDocuments]);

  // Polling for processing documents
  useEffect(() => {
    const processingDocs = documents.filter(d => d.status === 'pending' || d.status === 'processing');
    if (processingDocs.length > 0) {
      const interval = setInterval(() => {
        fetchDocuments();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [documents, fetchDocuments]);

  // Create knowledge base
  const handleCreateKB = async () => {
    if (!newKBName.trim()) {
      toast.error('Please enter a name for the knowledge base');
      return;
    }

    try {
      setIsCreating(true);
      const newKB = await api.post<KnowledgeBase>('/api/v1/kb', {
        name: newKBName,
        description: newKBDescription || null,
      });
      setKnowledgeBases(prev => [newKB, ...prev]);
      setSelectedKB(newKB);
      setShowCreateDialog(false);
      setNewKBName('');
      setNewKBDescription('');
      toast.success('Knowledge base created successfully');
    } catch (error: any) {
      toast.error(error.detail || 'Failed to create knowledge base');
    } finally {
      setIsCreating(false);
    }
  };

  // Delete knowledge base
  const handleDeleteKB = async (kb: KnowledgeBase) => {
    if (!confirm(`Are you sure you want to delete "${kb.name}"? This will delete all documents and cannot be undone.`)) {
      return;
    }

    try {
      await api.delete(`/api/v1/kb/${kb.id}`);
      setKnowledgeBases(prev => prev.filter(k => k.id !== kb.id));
      if (selectedKB?.id === kb.id) {
        setSelectedKB(knowledgeBases.find(k => k.id !== kb.id) || null);
      }
      toast.success('Knowledge base deleted');
    } catch (error: any) {
      toast.error(error.detail || 'Failed to delete knowledge base');
    }
  };

  // File upload with dropzone
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!selectedKB) {
      toast.error('Please select a knowledge base first');
      return;
    }

    setIsUploading(true);
    let successCount = 0;
    let errorCount = 0;

    for (const file of acceptedFiles) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('title', file.name);

        await api.upload(`/api/v1/kb/${selectedKB.id}/documents`, formData);
        successCount++;
      } catch (error: any) {
        errorCount++;
        console.error('Upload failed:', error);
      }
    }

    setIsUploading(false);
    await fetchDocuments();

    if (successCount > 0) {
      toast.success(`${successCount} file(s) uploaded successfully`);
    }
    if (errorCount > 0) {
      toast.error(`${errorCount} file(s) failed to upload`);
    }
  }, [selectedKB, fetchDocuments]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  // Delete document
  const handleDeleteDocument = async (doc: Document) => {
    if (!selectedKB) return;

    try {
      await api.delete(`/api/v1/kb/${selectedKB.id}/documents/${doc.id}`);
      setDocuments(prev => prev.filter(d => d.id !== doc.id));
      toast.success('Document deleted');
    } catch (error: any) {
      toast.error(error.detail || 'Failed to delete document');
    }
  };

  // Search
  const handleSearch = async () => {
    if (!selectedKB || !searchQuery.trim()) return;

    try {
      setIsSearching(true);
      const result = await api.post<{ query: string; results: SearchResult[]; total_results: number }>(
        `/api/v1/kb/${selectedKB.id}/search`,
        { query: searchQuery, top_k: 10 }
      );
      setSearchResults(result.results);
    } catch (error: any) {
      toast.error(error.detail || 'Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map(i => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Knowledge Base</h1>
          <p className="text-gray-600 mt-1">Upload documents to train your AI agent with your business knowledge.</p>
        </div>
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Knowledge Base
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Knowledge Base</DialogTitle>
              <DialogDescription>
                Create a new knowledge base to organize your documents.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <label className="text-sm font-medium">Name</label>
                <Input
                  placeholder="e.g., Product FAQs, Company Policies"
                  value={newKBName}
                  onChange={e => setNewKBName(e.target.value)}
                />
              </div>
              <div>
                <label className="text-sm font-medium">Description (optional)</label>
                <Textarea
                  placeholder="Describe what this knowledge base contains..."
                  value={newKBDescription}
                  onChange={e => setNewKBDescription(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateKB} disabled={isCreating}>
                {isCreating && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Empty state */}
      {knowledgeBases.length === 0 ? (
        <Card className="text-center py-12">
          <CardContent>
            <Database className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Knowledge Bases Yet</h3>
            <p className="text-gray-500 mb-4">
              Create your first knowledge base to start uploading documents.
            </p>
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Knowledge Base
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 lg:grid-cols-4">
          {/* Sidebar - Knowledge Base List */}
          <div className="lg:col-span-1 space-y-3">
            <h3 className="font-semibold text-sm text-gray-500 uppercase tracking-wider">
              Your Knowledge Bases
            </h3>
            {knowledgeBases.map(kb => (
              <Card
                key={kb.id}
                className={`cursor-pointer transition-all ${
                  selectedKB?.id === kb.id
                    ? 'ring-2 ring-primary border-primary'
                    : 'hover:border-primary/50'
                }`}
                onClick={() => setSelectedKB(kb)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <BookOpen className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h4 className="font-medium">{kb.name}</h4>
                        <p className="text-sm text-gray-500">
                          {kb.document_count} docs • {kb.chunk_count} chunks
                        </p>
                      </div>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={e => {
                            e.stopPropagation();
                            handleDeleteKB(kb);
                          }}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Main content */}
          <div className="lg:col-span-3 space-y-6">
            {selectedKB && (
              <>
                {/* Upload area */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Upload className="h-5 w-5" />
                      Upload Documents
                    </CardTitle>
                    <CardDescription>
                      Drag and drop files or click to browse. Supported: PDF, DOCX, TXT, MD
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div
                      {...getRootProps()}
                      className={`
                        border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                        ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-200 hover:border-primary/50'}
                        ${isUploading ? 'pointer-events-none opacity-50' : ''}
                      `}
                    >
                      <input {...getInputProps()} />
                      {isUploading ? (
                        <div className="flex flex-col items-center">
                          <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
                          <p className="text-sm text-gray-500">Uploading...</p>
                        </div>
                      ) : isDragActive ? (
                        <div className="flex flex-col items-center">
                          <Upload className="h-8 w-8 text-primary mb-2" />
                          <p className="text-sm text-primary font-medium">Drop files here...</p>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center">
                          <FolderOpen className="h-8 w-8 text-gray-400 mb-2" />
                          <p className="text-sm text-gray-600 mb-1">
                            Drag & drop files here, or <span className="text-primary font-medium">click to browse</span>
                          </p>
                          <p className="text-xs text-gray-400">Max 50MB per file</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Search */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Search className="h-5 w-5" />
                      Search Knowledge Base
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex gap-2">
                      <Input
                        placeholder="Ask a question to test the knowledge base..."
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleSearch()}
                      />
                      <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
                        {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                      </Button>
                    </div>

                    {searchResults.length > 0 && (
                      <div className="mt-4 space-y-3">
                        <h4 className="font-medium text-sm text-gray-500">Search Results</h4>
                        {searchResults.map((result, idx) => (
                          <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-medium">{result.document_title}</span>
                              <Badge variant="outline">{(result.score * 100).toFixed(1)}% match</Badge>
                            </div>
                            <p className="text-sm text-gray-600 line-clamp-3">{result.content}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Documents list */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>Documents</CardTitle>
                      <Button variant="ghost" size="sm" onClick={fetchDocuments}>
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Refresh
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {documents.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">
                        <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p>No documents uploaded yet</p>
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Document</TableHead>
                            <TableHead>Size</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>Chunks</TableHead>
                            <TableHead className="w-[50px]"></TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {documents.map(doc => (
                            <TableRow key={doc.id}>
                              <TableCell>
                                <div className="flex items-center gap-3">
                                  {getFileIcon(doc.file_type)}
                                  <div>
                                    <p className="font-medium">{doc.title}</p>
                                    <p className="text-xs text-gray-500">{doc.file_name}</p>
                                  </div>
                                </div>
                              </TableCell>
                              <TableCell className="text-sm text-gray-500">
                                {formatFileSize(doc.file_size)}
                              </TableCell>
                              <TableCell>{getStatusBadge(doc.status)}</TableCell>
                              <TableCell className="text-sm text-gray-500">
                                {doc.chunk_count ?? '-'}
                              </TableCell>
                              <TableCell>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8 text-destructive hover:text-destructive"
                                  onClick={() => handleDeleteDocument(doc)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
