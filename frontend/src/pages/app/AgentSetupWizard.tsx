import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { toast } from 'sonner';
import {
  Loader2,
  Building2,
  Languages,
  Package,
  MapPin,
  Gift,
  Shield,
  CheckCircle2,
  Plus,
  Trash2,
  BookOpen,
  ExternalLink,
  Clock,
  DollarSign,
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
  CardFooter,
} from '@/components/ui/card';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
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
  WizardProvider,
  useWizard,
  WizardSteps,
  WizardNavigation,
  WizardStepContent,
} from '@/components/wizard';
import { useCustomers } from '@/hooks/useAgent';
import { api } from '@/lib/api';

// ===== Types =====
interface ServiceItem {
  id?: number;
  name: string;
  description?: string;
  price?: number;
  duration_minutes?: number;
  warranty_period?: string;
}

interface ProductItem {
  id?: number;
  name: string;
  description?: string;
  price_min?: number;
  price_max?: number;
  warranty_period?: string;
  shipping_available?: boolean;
}

interface LocationItem {
  id?: number;
  name: string;
  address: string;
  city?: string;
  country?: string;
  phone?: string;
  working_hours?: string;
  is_headquarters?: boolean;
}

interface PromotionItem {
  id?: number;
  name: string;
  description?: string;
  discount_type?: 'percentage' | 'fixed';
  discount_value?: number;
  valid_from?: string;
  valid_until?: string;
  promo_code?: string;
  is_active?: boolean;
}

interface WizardFormData {
  // Step 1: Business Profile
  business_name: string;
  business_description: string;
  supported_language: 'en' | 'ar' | 'both';
  tone: 'professional' | 'friendly' | 'casual' | 'formal';
  website_url?: string;

  // Step 2: Business Type & Services
  business_type: 'Retail' | 'Service' | 'Both';
  services: ServiceItem[];
  products: ProductItem[];

  // Step 3: Knowledge Base
  knowledge_base_ids: number[];

  // Step 4: Locations
  locations: LocationItem[];
  booking_enabled: boolean;
  booking_link?: string;

  // Step 5: Promotions
  promotions: PromotionItem[];

  // Step 6: Policies & Advanced
  business_policies: string;
  custom_instructions?: string;
}

// Wizard steps configuration
const WIZARD_STEPS = [
  { id: 'profile', title: 'Business Profile', icon: Building2 },
  { id: 'services', title: 'Services & Products', icon: Package },
  { id: 'knowledge', title: 'Knowledge Base', icon: BookOpen },
  { id: 'locations', title: 'Locations & Booking', icon: MapPin },
  { id: 'promotions', title: 'Promotions', icon: Gift },
  { id: 'policies', title: 'Policies & Advanced', icon: Shield },
];

// ===== Wizard Steps Components =====

function Step1BusinessProfile() {
  const { data, updateData } = useWizard();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Building2 className="h-5 w-5" />
          Business Profile
        </CardTitle>
        <CardDescription>
          Tell us about your business so the AI can represent you accurately.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Business Name *</label>
          <Input
            placeholder="e.g., Genius AI"
            value={data.business_name || ''}
            onChange={e => updateData({ business_name: e.target.value })}
          />
          <p className="text-xs text-gray-500">The AI will use this name when greeting customers.</p>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Business Description</label>
          <Textarea
            placeholder="Describe what your business does, what makes you special, and your main value proposition..."
            className="min-h-[100px]"
            value={data.business_description || ''}
            onChange={e => updateData({ business_description: e.target.value })}
          />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium">Primary Language *</label>
            <Select
              value={data.supported_language || 'both'}
              onValueChange={value => updateData({ supported_language: value as any })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English Only</SelectItem>
                <SelectItem value="ar">العربية فقط (Arabic Only)</SelectItem>
                <SelectItem value="both">Bilingual (English + Arabic)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-gray-500">The AI will respond in this language.</p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Communication Tone</label>
            <Select
              value={data.tone || 'professional'}
              onValueChange={value => updateData({ tone: value as any })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select tone" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="professional">Professional (Courteous & Formal)</SelectItem>
                <SelectItem value="friendly">Friendly (Warm & Approachable)</SelectItem>
                <SelectItem value="casual">Casual (Relaxed & Conversational)</SelectItem>
                <SelectItem value="formal">Formal (Strictly Business)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Website URL</label>
          <Input
            placeholder="https://www.yourwebsite.com"
            type="url"
            value={data.website_url || ''}
            onChange={e => updateData({ website_url: e.target.value })}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function Step2ServicesProducts() {
  const { data, updateData } = useWizard();

  const services: ServiceItem[] = data.services || [];
  const products: ProductItem[] = data.products || [];
  const businessType = data.business_type || 'Service';

  // Determine which tables to show based on business type
  const showServices = businessType === 'Service' || businessType === 'Both';
  const showProducts = businessType === 'Retail' || businessType === 'Both';

  // Helper to create an empty service row
  const createEmptyService = (): ServiceItem => ({
    name: '',
    description: '',
    price: undefined,
    duration_minutes: undefined,
  });

  // Helper to create an empty product row
  const createEmptyProduct = (): ProductItem => ({
    name: '',
    description: '',
    price_min: undefined,
    price_max: undefined,
  });

  // Initialize with one empty row if needed
  const servicesWithEmpty = [...services];
  if (showServices && (servicesWithEmpty.length === 0 || servicesWithEmpty[servicesWithEmpty.length - 1]?.name?.trim())) {
    servicesWithEmpty.push(createEmptyService());
  }

  const productsWithEmpty = [...products];
  if (showProducts && (productsWithEmpty.length === 0 || productsWithEmpty[productsWithEmpty.length - 1]?.name?.trim())) {
    productsWithEmpty.push(createEmptyProduct());
  }

  const handleServiceChange = (index: number, field: keyof ServiceItem, value: any) => {
    const updatedServices = [...services];

    // If editing the empty row (last row), add it to the actual services array
    if (index >= services.length) {
      updatedServices.push(createEmptyService());
    }

    updatedServices[index] = { ...updatedServices[index], [field]: value };

    // Filter out completely empty rows except the last one
    const filteredServices = updatedServices.filter((s, i) =>
      i === updatedServices.length - 1 || s.name?.trim()
    );

    updateData({ services: filteredServices });
  };

  const handleDeleteService = (index: number) => {
    const updatedServices = services.filter((_, i) => i !== index);
    updateData({ services: updatedServices });
  };

  const handleProductChange = (index: number, field: keyof ProductItem, value: any) => {
    const updatedProducts = [...products];

    // If editing the empty row (last row), add it to the actual products array
    if (index >= products.length) {
      updatedProducts.push(createEmptyProduct());
    }

    updatedProducts[index] = { ...updatedProducts[index], [field]: value };

    // Filter out completely empty rows except the last one
    const filteredProducts = updatedProducts.filter((p, i) =>
      i === updatedProducts.length - 1 || p.name?.trim()
    );

    updateData({ products: filteredProducts });
  };

  const handleDeleteProduct = (index: number) => {
    const updatedProducts = products.filter((_, i) => i !== index);
    updateData({ products: updatedProducts });
  };

  return (
    <div className="space-y-6">
      {/* Business Type */}
      <Card>
        <CardHeader>
          <CardTitle>Business Type</CardTitle>
          <CardDescription>Select your business type to see the relevant input fields.</CardDescription>
        </CardHeader>
        <CardContent>
          <Select
            value={businessType}
            onValueChange={value => updateData({ business_type: value as any })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select business type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Retail">Retail Store (Products only)</SelectItem>
              <SelectItem value="Service">Service Business (Services only)</SelectItem>
              <SelectItem value="Both">Both (Services & Products)</SelectItem>
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      {/* Services - Only show if Service or Both */}
      {showServices && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5" />
              Services
            </CardTitle>
            <CardDescription>Add your services with pricing and duration. A new row appears when you fill the current one.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[30%]">Service Name *</TableHead>
                  <TableHead className="w-[25%]">Description</TableHead>
                  <TableHead className="w-[15%]">Price ($)</TableHead>
                  <TableHead className="w-[15%]">Duration (min)</TableHead>
                  <TableHead className="w-[15%]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {servicesWithEmpty.map((service, idx) => {
                  const isLastRow = idx === servicesWithEmpty.length - 1 && !service.name?.trim();
                  const isFilledRow = service.name?.trim();

                  return (
                    <TableRow key={idx} className={isLastRow ? 'bg-gray-50/50' : ''}>
                      <TableCell>
                        <Input
                          placeholder="e.g., Deep Tissue Massage"
                          value={service.name || ''}
                          onChange={e => handleServiceChange(idx, 'name', e.target.value)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        <Input
                          placeholder="Brief description..."
                          value={service.description || ''}
                          onChange={e => handleServiceChange(idx, 'description', e.target.value)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        <Input
                          type="number"
                          placeholder="0.00"
                          value={service.price || ''}
                          onChange={e => handleServiceChange(idx, 'price', e.target.value ? parseFloat(e.target.value) : undefined)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        <Input
                          type="number"
                          placeholder="60"
                          value={service.duration_minutes || ''}
                          onChange={e => handleServiceChange(idx, 'duration_minutes', e.target.value ? parseInt(e.target.value) : undefined)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        {isFilledRow && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                            onClick={() => handleDeleteService(idx)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            {services.filter(s => s.name?.trim()).length > 0 && (
              <p className="text-xs text-gray-500 mt-2">
                {services.filter(s => s.name?.trim()).length} service(s) added
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Products - Only show if Retail or Both */}
      {showProducts && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              Products
            </CardTitle>
            <CardDescription>Add products you sell. A new row appears when you fill the current one.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[30%]">Product Name *</TableHead>
                  <TableHead className="w-[25%]">Description</TableHead>
                  <TableHead className="w-[15%]">Min Price ($)</TableHead>
                  <TableHead className="w-[15%]">Max Price ($)</TableHead>
                  <TableHead className="w-[15%]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {productsWithEmpty.map((product, idx) => {
                  const isLastRow = idx === productsWithEmpty.length - 1 && !product.name?.trim();
                  const isFilledRow = product.name?.trim();

                  return (
                    <TableRow key={idx} className={isLastRow ? 'bg-gray-50/50' : ''}>
                      <TableCell>
                        <Input
                          placeholder="e.g., Organic Face Serum"
                          value={product.name || ''}
                          onChange={e => handleProductChange(idx, 'name', e.target.value)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        <Input
                          placeholder="Brief description..."
                          value={product.description || ''}
                          onChange={e => handleProductChange(idx, 'description', e.target.value)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        <Input
                          type="number"
                          placeholder="0.00"
                          value={product.price_min || ''}
                          onChange={e => handleProductChange(idx, 'price_min', e.target.value ? parseFloat(e.target.value) : undefined)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        <Input
                          type="number"
                          placeholder="0.00"
                          value={product.price_max || ''}
                          onChange={e => handleProductChange(idx, 'price_max', e.target.value ? parseFloat(e.target.value) : undefined)}
                          className="h-9"
                        />
                      </TableCell>
                      <TableCell>
                        {isFilledRow && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
                            onClick={() => handleDeleteProduct(idx)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            {products.filter(p => p.name?.trim()).length > 0 && (
              <p className="text-xs text-gray-500 mt-2">
                {products.filter(p => p.name?.trim()).length} product(s) added
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ServiceForm({
  initial,
  onSave,
  onCancel,
}: {
  initial: ServiceItem | null;
  onSave: (service: ServiceItem) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name || '');
  const [description, setDescription] = useState(initial?.description || '');
  const [price, setPrice] = useState(initial?.price?.toString() || '');
  const [duration, setDuration] = useState(initial?.duration_minutes?.toString() || '');
  const [warranty, setWarranty] = useState(initial?.warranty_period || '');

  const handleSubmit = () => {
    if (!name.trim()) {
      toast.error('Service name is required');
      return;
    }
    onSave({
      name,
      description: description || undefined,
      price: price ? parseFloat(price) : undefined,
      duration_minutes: duration ? parseInt(duration) : undefined,
      warranty_period: warranty || undefined,
    });
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>{initial ? 'Edit Service' : 'Add Service'}</DialogTitle>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Service Name *</label>
          <Input placeholder="e.g., Deep Tissue Massage" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Description</label>
          <Textarea placeholder="Describe the service..." value={description} onChange={e => setDescription(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Price ($)</label>
            <Input type="number" placeholder="0.00" value={price} onChange={e => setPrice(e.target.value)} />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Duration (minutes)</label>
            <Input type="number" placeholder="60" value={duration} onChange={e => setDuration(e.target.value)} />
          </div>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Warranty Period</label>
          <Input placeholder="e.g., 30 days" value={warranty} onChange={e => setWarranty(e.target.value)} />
        </div>
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={handleSubmit}>{initial ? 'Update' : 'Add'}</Button>
      </DialogFooter>
    </>
  );
}

function ProductForm({
  initial,
  onSave,
  onCancel,
}: {
  initial: ProductItem | null;
  onSave: (product: ProductItem) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name || '');
  const [description, setDescription] = useState(initial?.description || '');
  const [priceMin, setPriceMin] = useState(initial?.price_min?.toString() || '');
  const [priceMax, setPriceMax] = useState(initial?.price_max?.toString() || '');
  const [shipping, setShipping] = useState(initial?.shipping_available || false);

  const handleSubmit = () => {
    if (!name.trim()) {
      toast.error('Product name is required');
      return;
    }
    onSave({
      name,
      description: description || undefined,
      price_min: priceMin ? parseFloat(priceMin) : undefined,
      price_max: priceMax ? parseFloat(priceMax) : undefined,
      shipping_available: shipping,
    });
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>{initial ? 'Edit Product' : 'Add Product'}</DialogTitle>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Product Name *</label>
          <Input placeholder="e.g., Organic Face Serum" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Description</label>
          <Textarea placeholder="Describe the product..." value={description} onChange={e => setDescription(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Min Price ($)</label>
            <Input type="number" placeholder="0.00" value={priceMin} onChange={e => setPriceMin(e.target.value)} />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Max Price ($)</label>
            <Input type="number" placeholder="0.00" value={priceMax} onChange={e => setPriceMax(e.target.value)} />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Switch checked={shipping} onCheckedChange={setShipping} />
          <label className="text-sm">Shipping available</label>
        </div>
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={handleSubmit}>{initial ? 'Update' : 'Add'}</Button>
      </DialogFooter>
    </>
  );
}

function Step3KnowledgeBase() {
  const { data } = useWizard();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          Knowledge Base
        </CardTitle>
        <CardDescription>
          Connect your knowledge base to enable RAG (Retrieval Augmented Generation).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
          <p className="text-sm text-blue-800">
            You can upload documents in the{' '}
            <Link to="/app/knowledge-base" className="font-medium underline">
              Knowledge Base section
            </Link>
            . Documents uploaded there will be used by the AI to answer questions.
          </p>
        </div>

        <div className="text-center py-8">
          <BookOpen className="h-12 w-12 mx-auto text-gray-400 mb-4" />
          <h3 className="font-medium text-gray-900 mb-2">Knowledge Base Integration</h3>
          <p className="text-sm text-gray-500 mb-4">
            Upload PDFs, DOCX, TXT files with FAQs, policies, and product info.
          </p>
          <Link to="/app/knowledge-base">
            <Button variant="outline">
              <ExternalLink className="h-4 w-4 mr-2" />
              Go to Knowledge Base
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}

function Step4Locations() {
  const { data, updateData } = useWizard();
  const [showLocationDialog, setShowLocationDialog] = useState(false);
  const [editingLocation, setEditingLocation] = useState<LocationItem | null>(null);

  const locations: LocationItem[] = data.locations || [];

  const handleSaveLocation = (location: LocationItem) => {
    if (editingLocation) {
      updateData({
        locations: locations.map(l => (l === editingLocation ? location : l)),
      });
    } else {
      updateData({ locations: [...locations, location] });
    }
    setShowLocationDialog(false);
    setEditingLocation(null);
  };

  const handleDeleteLocation = (location: LocationItem) => {
    updateData({ locations: locations.filter(l => l !== location) });
  };

  return (
    <div className="space-y-6">
      {/* Locations */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="h-5 w-5" />
                Business Locations
              </CardTitle>
              <CardDescription>Add your branches or service locations (optional).</CardDescription>
            </div>
            <Dialog open={showLocationDialog} onOpenChange={setShowLocationDialog}>
              <DialogTrigger asChild>
                <Button onClick={() => setEditingLocation(null)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add Location
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg">
                <LocationForm
                  initial={editingLocation}
                  onSave={handleSaveLocation}
                  onCancel={() => setShowLocationDialog(false)}
                />
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          {locations.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <MapPin className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No locations added yet (optional)</p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {locations.map((loc, idx) => (
                <Card key={idx} className="relative">
                  <CardContent className="pt-4">
                    {loc.is_headquarters && (
                      <Badge className="absolute top-2 right-2" variant="secondary">HQ</Badge>
                    )}
                    <h4 className="font-medium">{loc.name}</h4>
                    <p className="text-sm text-gray-500">{loc.address}</p>
                    {loc.city && <p className="text-sm text-gray-500">{loc.city}, {loc.country}</p>}
                    {loc.phone && <p className="text-sm text-gray-500">{loc.phone}</p>}
                    {loc.working_hours && (
                      <p className="text-xs text-gray-400 mt-1">
                        <Clock className="h-3 w-3 inline mr-1" />
                        {loc.working_hours}
                      </p>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute bottom-2 right-2 text-destructive"
                      onClick={() => handleDeleteLocation(loc)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Booking */}
      <Card>
        <CardHeader>
          <CardTitle>Online Booking</CardTitle>
          <CardDescription>Enable customers to book appointments via an external link.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Enable Booking Link</p>
              <p className="text-sm text-gray-500">Allow AI to share your booking link with customers.</p>
            </div>
            <Switch
              checked={data.booking_enabled || false}
              onCheckedChange={checked => updateData({ booking_enabled: checked })}
            />
          </div>

          {data.booking_enabled && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Booking URL</label>
              <Input
                placeholder="https://calendly.com/your-business or https://cal.com/your-link"
                value={data.booking_link || ''}
                onChange={e => updateData({ booking_link: e.target.value })}
              />
              <p className="text-xs text-gray-500">
                Paste your Calendly, Cal.com, or other scheduling link.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function LocationForm({
  initial,
  onSave,
  onCancel,
}: {
  initial: LocationItem | null;
  onSave: (location: LocationItem) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name || '');
  const [address, setAddress] = useState(initial?.address || '');
  const [city, setCity] = useState(initial?.city || '');
  const [country, setCountry] = useState(initial?.country || '');
  const [phone, setPhone] = useState(initial?.phone || '');
  const [hours, setHours] = useState(initial?.working_hours || '');
  const [isHQ, setIsHQ] = useState(initial?.is_headquarters || false);

  const handleSubmit = () => {
    if (!name.trim() || !address.trim()) {
      toast.error('Name and address are required');
      return;
    }
    onSave({
      name,
      address,
      city: city || undefined,
      country: country || undefined,
      phone: phone || undefined,
      working_hours: hours || undefined,
      is_headquarters: isHQ,
    });
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>{initial ? 'Edit Location' : 'Add Location'}</DialogTitle>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Location Name *</label>
          <Input placeholder="e.g., Downtown Branch" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Address *</label>
          <Input placeholder="123 Main Street" value={address} onChange={e => setAddress(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">City</label>
            <Input placeholder="Riyadh" value={city} onChange={e => setCity(e.target.value)} />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Country</label>
            <Input placeholder="Saudi Arabia" value={country} onChange={e => setCountry(e.target.value)} />
          </div>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Phone</label>
          <Input placeholder="+966 xxx xxx xxxx" value={phone} onChange={e => setPhone(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Working Hours</label>
          <Input placeholder="9 AM - 9 PM, Sat-Thu" value={hours} onChange={e => setHours(e.target.value)} />
        </div>
        <div className="flex items-center gap-2">
          <Switch checked={isHQ} onCheckedChange={setIsHQ} />
          <label className="text-sm">This is the headquarters</label>
        </div>
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={handleSubmit}>{initial ? 'Update' : 'Add'}</Button>
      </DialogFooter>
    </>
  );
}

function Step5Promotions() {
  const { data, updateData } = useWizard();
  const [showPromoDialog, setShowPromoDialog] = useState(false);
  const [editingPromo, setEditingPromo] = useState<PromotionItem | null>(null);

  const promotions: PromotionItem[] = data.promotions || [];

  const handleSavePromo = (promo: PromotionItem) => {
    if (editingPromo) {
      updateData({
        promotions: promotions.map(p => (p === editingPromo ? promo : p)),
      });
    } else {
      updateData({ promotions: [...promotions, promo] });
    }
    setShowPromoDialog(false);
    setEditingPromo(null);
  };

  const handleDeletePromo = (promo: PromotionItem) => {
    updateData({ promotions: promotions.filter(p => p !== promo) });
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Gift className="h-5 w-5" />
              Promotions & Discounts
            </CardTitle>
            <CardDescription>Add current promotions the AI can tell customers about.</CardDescription>
          </div>
          <Dialog open={showPromoDialog} onOpenChange={setShowPromoDialog}>
            <DialogTrigger asChild>
              <Button onClick={() => setEditingPromo(null)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Promotion
              </Button>
            </DialogTrigger>
            <DialogContent>
              <PromotionForm
                initial={editingPromo}
                onSave={handleSavePromo}
                onCancel={() => setShowPromoDialog(false)}
              />
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        {promotions.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Gift className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No promotions added yet (optional)</p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {promotions.map((promo, idx) => (
              <Card key={idx} className="relative">
                <CardContent className="pt-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium">{promo.name}</h4>
                      {promo.description && (
                        <p className="text-sm text-gray-500">{promo.description}</p>
                      )}
                      {promo.discount_value && (
                        <Badge variant="secondary" className="mt-2">
                          {promo.discount_type === 'percentage'
                            ? `${promo.discount_value}% OFF`
                            : `$${promo.discount_value} OFF`}
                        </Badge>
                      )}
                      {promo.promo_code && (
                        <p className="text-xs text-gray-400 mt-1">
                          Code: <span className="font-mono">{promo.promo_code}</span>
                        </p>
                      )}
                      {promo.valid_until && (
                        <p className="text-xs text-gray-400">
                          Valid until: {new Date(promo.valid_until).toLocaleDateString()}
                        </p>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive"
                      onClick={() => handleDeletePromo(promo)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function PromotionForm({
  initial,
  onSave,
  onCancel,
}: {
  initial: PromotionItem | null;
  onSave: (promo: PromotionItem) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name || '');
  const [description, setDescription] = useState(initial?.description || '');
  const [discountType, setDiscountType] = useState<'percentage' | 'fixed'>(initial?.discount_type || 'percentage');
  const [discountValue, setDiscountValue] = useState(initial?.discount_value?.toString() || '');
  const [promoCode, setPromoCode] = useState(initial?.promo_code || '');
  const [validUntil, setValidUntil] = useState(initial?.valid_until || '');

  const handleSubmit = () => {
    if (!name.trim()) {
      toast.error('Promotion name is required');
      return;
    }
    onSave({
      name,
      description: description || undefined,
      discount_type: discountType,
      discount_value: discountValue ? parseFloat(discountValue) : undefined,
      promo_code: promoCode || undefined,
      valid_until: validUntil || undefined,
      is_active: true,
    });
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>{initial ? 'Edit Promotion' : 'Add Promotion'}</DialogTitle>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Promotion Name *</label>
          <Input placeholder="e.g., Summer Sale" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Description</label>
          <Textarea placeholder="Describe the promotion..." value={description} onChange={e => setDescription(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Discount Type</label>
            <Select value={discountType} onValueChange={v => setDiscountType(v as any)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="percentage">Percentage (%)</SelectItem>
                <SelectItem value="fixed">Fixed Amount ($)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Discount Value</label>
            <Input type="number" placeholder="20" value={discountValue} onChange={e => setDiscountValue(e.target.value)} />
          </div>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Promo Code</label>
          <Input placeholder="SUMMER20" value={promoCode} onChange={e => setPromoCode(e.target.value.toUpperCase())} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium">Valid Until</label>
          <Input type="date" value={validUntil} onChange={e => setValidUntil(e.target.value)} />
        </div>
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={onCancel}>Cancel</Button>
        <Button onClick={handleSubmit}>{initial ? 'Update' : 'Add'}</Button>
      </DialogFooter>
    </>
  );
}

function Step6PoliciesAdvanced() {
  const { data, updateData } = useWizard();

  return (
    <div className="space-y-6">
      {/* Business Policies */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Business Policies
          </CardTitle>
          <CardDescription>
            Define rules the AI must always follow. These are non-negotiable.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder={`Example policies:
- All bookings require a 50% deposit
- Cancellations must be made 24 hours in advance
- No refunds after service is provided
- Gift cards are non-refundable
- We do not offer discounts on already reduced items`}
            className="min-h-[200px]"
            value={data.business_policies || ''}
            onChange={e => updateData({ business_policies: e.target.value })}
          />
          <p className="text-xs text-gray-500 mt-2">
            The AI will strictly enforce these policies when answering customer questions.
          </p>
        </CardContent>
      </Card>

      {/* Advanced: Raw Instructions */}
      <Card>
        <CardHeader>
          <CardTitle>Advanced: Custom System Prompt</CardTitle>
          <CardDescription>
            For power users only. Override the auto-generated system prompt.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Leave empty to use the auto-generated prompt based on your settings above..."
            className="min-h-[150px] font-mono text-sm"
            value={data.custom_instructions || ''}
            onChange={e => updateData({ custom_instructions: e.target.value })}
          />
          <p className="text-xs text-amber-600 mt-2">
            ⚠️ If you enter a custom prompt, it will override all other settings (tone, language, policies, etc.)
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// ===== Main Wizard Component =====

function WizardContent() {
  const { currentStep, data, isSaving, setIsSaving, setData } = useWizard();
  const { data: customers } = useCustomers();
  const navigate = useNavigate();
  const customerId = customers?.[0]?.id;
  const [powerUserMode, setPowerUserMode] = useState(false);

  // Filter steps based on power user mode
  const visibleSteps = powerUserMode
    ? WIZARD_STEPS
    : WIZARD_STEPS.filter(s => s.id !== 'policies');

  const totalVisibleSteps = visibleSteps.length;

  // Load existing wizard config
  useEffect(() => {
    if (!customerId) return;

    const loadConfig = async () => {
      try {
        const response = await api.get<any>(`/api/v1/whatsapp/agents/wizard/${customerId}/details`);
        if (response) {
          setData({
            business_name: response.business_name || '',
            business_description: response.business_description || '',
            supported_language: response.supported_language || 'both',
            tone: response.tone || 'professional',
            website_url: response.website_url || '',
            business_type: response.business_type || 'service',
            services: response.services || [],
            products: response.products || [],
            knowledge_base_ids: response.knowledge_base_ids || [],
            locations: response.locations || [],
            booking_enabled: response.booking_enabled || false,
            booking_link: response.booking_link || '',
            promotions: response.promotions || [],
            business_policies: response.business_policies || '',
            custom_instructions: response.custom_instructions || '',
          });
        }
      } catch (error) {
        // No existing config, use customer name as default
        if (customers?.[0]?.name) {
          setData({ business_name: customers[0].name });
        }
      }
    };

    loadConfig();
  }, [customerId, customers, setData]);

  const handleSave = async () => {
    if (!customerId) {
      toast.error('No business found');
      return;
    }

    setIsSaving(true);
    try {
      await api.post(`/api/v1/whatsapp/agents/wizard`, {
        customer_id: customerId,
        ...data,
        wizard_completed: currentStep === totalVisibleSteps - 1,
        wizard_current_step: currentStep,
      });
      toast.success('Configuration saved successfully!');

      if (currentStep === totalVisibleSteps - 1) {
        toast.success('Wizard completed! Your AI agent is now configured.');
        navigate('/app/agent-setup');
      }
    } catch (error: any) {
      toast.error(error.detail || 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI Agent Setup Wizard</h1>
          <p className="text-gray-600 mt-1">Configure your AI assistant step by step.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <Switch
              id="power-user-mode"
              checked={powerUserMode}
              onCheckedChange={setPowerUserMode}
            />
            <label
              htmlFor="power-user-mode"
              className="text-gray-600 cursor-pointer select-none"
            >
              Power User Mode
            </label>
          </div>
        </div>
      </div>

      {/* Steps indicator */}
      <WizardSteps
        steps={visibleSteps.map((s, i) => ({
          id: i,
          title: s.title,
          icon: s.icon,
        }))}
      />

      {/* Step content */}
      <WizardStepContent step={0}><Step1BusinessProfile /></WizardStepContent>
      <WizardStepContent step={1}><Step2ServicesProducts /></WizardStepContent>
      <WizardStepContent step={2}><Step3KnowledgeBase /></WizardStepContent>
      <WizardStepContent step={3}><Step4Locations /></WizardStepContent>
      <WizardStepContent step={4}><Step5Promotions /></WizardStepContent>
      {powerUserMode && (
        <WizardStepContent step={5}><Step6PoliciesAdvanced /></WizardStepContent>
      )}

      {/* Navigation */}
      <WizardNavigation
        totalSteps={totalVisibleSteps}
        onSave={handleSave}
        canProceed={!!data.business_name?.trim()}
      />
    </div>
  );
}

export default function AgentSetupWizard() {
  const { data: customers, isLoading } = useCustomers();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!customers || customers.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <Card className="w-full max-w-md text-center">
          <CardHeader>
            <CardTitle>No Business Found</CardTitle>
            <CardDescription>
              You need to create a business profile before setting up an AI agent.
            </CardDescription>
          </CardHeader>
          <CardFooter className="justify-center">
            <Link to="/app/businesses">
              <Button variant="outline">Create Business</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <WizardProvider totalSteps={WIZARD_STEPS.length}>
      <WizardContent />
    </WizardProvider>
  );
}
