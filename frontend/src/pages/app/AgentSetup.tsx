import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Loader2, Save, Bot, Power, PauseCircle, PlayCircle, AlertCircle, Wand2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";

import { useCustomers, useAgentConfig, useUpdateAgentConfig, useAIControl } from "@/hooks/useAgent";

const formSchema = z.object({
  business_name: z.string().min(2, "Business name must be at least 2 characters"),
  tone: z.enum(["professional", "friendly", "casual", "formal"]),
  behavior_rules: z.string().optional(),
  custom_instructions: z.string().optional(),
  ai_pause_duration_minutes: z.coerce.number().min(5).max(1440),
});

export default function AgentSetup() {
  const [selectedCustomerId, setSelectedCustomerId] = useState<number | null>(null);
  
  // Fetch customers
  const { data: customers, isLoading: isLoadingCustomers } = useCustomers();
  
  // Set default customer
  useEffect(() => {
    if (customers && customers.length > 0 && !selectedCustomerId) {
      setSelectedCustomerId(customers[0].id);
    }
  }, [customers, selectedCustomerId]);

  // Fetch agent config
  const { data: agentConfig, isLoading: isLoadingConfig, error: configError } = useAgentConfig(selectedCustomerId || undefined);
  
  // Mutations
  const updateConfigMutation = useUpdateAgentConfig();
  const aiControlMutation = useAIControl();

  // Form setup
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      business_name: "",
      tone: "professional",
      behavior_rules: "",
      custom_instructions: "",
      ai_pause_duration_minutes: 30,
    },
  });

  // Update form when config loads
  useEffect(() => {
    if (agentConfig) {
      form.reset({
        business_name: agentConfig.business_name,
        tone: agentConfig.tone,
        behavior_rules: agentConfig.behavior_rules || "",
        custom_instructions: agentConfig.custom_instructions || "",
        ai_pause_duration_minutes: agentConfig.ai_pause_duration_minutes,
      });
    } else if (customers && customers.length > 0) {
      // Pre-fill business name from customer name if no config exists
      const customer = customers.find(c => c.id === selectedCustomerId);
      if (customer) {
        form.setValue("business_name", customer.name);
      }
    }
  }, [agentConfig, customers, selectedCustomerId, form]);

  const onSubmit = (values: z.infer<typeof formSchema>) => {
    if (!selectedCustomerId) return;

    updateConfigMutation.mutate(
      {
        customer_id: selectedCustomerId,
        business_name: values.business_name,
        tone: values.tone,
        behavior_rules: values.behavior_rules,
        custom_instructions: values.custom_instructions,
        ai_pause_duration_minutes: values.ai_pause_duration_minutes,
      },
      {
        onSuccess: () => {
          toast.success("Agent configuration saved successfully");
        },
        onError: (error: any) => {
          toast.error(error.detail || "Failed to save configuration");
        },
      }
    );
  };

  const handleAIControl = (action: 'pause' | 'resume' | 'toggle') => {
    if (!selectedCustomerId) return;
    
    aiControlMutation.mutate(
      { customerId: selectedCustomerId, data: { action } },
      {
        onSuccess: (data) => {
          toast.success(data.message);
        },
        onError: () => {
          toast.error("Failed to update AI status");
        },
      }
    );
  };

  if (isLoadingCustomers) {
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
              You need to create a business profile before configuring an AI agent.
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

  const isAIActive = agentConfig?.ai_enabled && !agentConfig?.ai_pause_until;
  const isPaused = agentConfig?.ai_pause_until && new Date(agentConfig.ai_pause_until) > new Date();

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI Agent Configuration</h1>
          <p className="text-gray-600 mt-1">Customize your AI assistant's personality and behavior</p>
        </div>
        
        <div className="flex items-center gap-2">
          <Link to="/app/agent-wizard">
            <Button variant="outline" className="gap-2">
              <Wand2 className="h-4 w-4" />
              Setup Wizard
            </Button>
          </Link>
          
          {/* Customer Selector (if multiple) */}
          {customers.length > 1 && selectedCustomerId && (
            <Select 
              value={selectedCustomerId.toString()} 
              onValueChange={(val) => setSelectedCustomerId(parseInt(val))}
            >
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Select Business" />
              </SelectTrigger>
              <SelectContent>
                {customers.map((c) => (
                  <SelectItem key={c.id} value={c.id.toString()}>
                    {c.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </div>

      {/* Wizard Prompt Banner */}
      {!agentConfig?.wizard_completed && (
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                <Wand2 className="h-6 w-6" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-lg text-blue-900">Complete the Setup Wizard</h3>
                <p className="text-blue-700 text-sm">
                  Use our step-by-step wizard to configure services, locations, promotions, and more.
                </p>
              </div>
              <Link to="/app/agent-wizard">
                <Button>
                  <Wand2 className="h-4 w-4 mr-2" />
                  Start Wizard
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Status Card */}
      <Card className="bg-gradient-to-r from-slate-50 to-white border-l-4 border-l-primary">
        <CardContent className="pt-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-full ${isAIActive ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-500'}`}>
              <Bot className="h-6 w-6" />
            </div>
            <div>
              <h3 className="font-semibold text-lg flex items-center gap-2">
                AI Status: 
                <span className={isAIActive ? "text-green-600" : "text-gray-500"}>
                  {isAIActive ? "Active" : isPaused ? "Paused" : "Disabled"}
                </span>
                {isPaused && (
                  <Badge variant="outline" className="text-amber-600 border-amber-200 bg-amber-50">
                    Until {new Date(agentConfig!.ai_pause_until!).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                  </Badge>
                )}
              </h3>
              <p className="text-sm text-gray-500">
                {isAIActive 
                  ? "Your AI agent is currently responding to customer messages." 
                  : "Your AI agent is not sending automated responses."}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isPaused ? (
              <Button variant="outline" size="sm" onClick={() => handleAIControl('resume')}>
                <PlayCircle className="h-4 w-4 mr-2" />
                Resume
              </Button>
            ) : (
              <Button variant="outline" size="sm" onClick={() => handleAIControl('pause')} disabled={!agentConfig?.ai_enabled}>
                <PauseCircle className="h-4 w-4 mr-2" />
                Pause 30m
              </Button>
            )}
            <div className="flex items-center gap-2 ml-2 border-l pl-4">
              <Switch 
                checked={agentConfig?.ai_enabled || false}
                onCheckedChange={() => handleAIControl('toggle')}
              />
              <span className="text-sm font-medium">Enable AI</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <Tabs defaultValue="basic" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="basic">Basic Settings</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>
            
            <TabsContent value="basic" className="space-y-4 mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Personality Profile</CardTitle>
                  <CardDescription>Define how your AI agent presents itself to customers.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <FormField
                    control={form.control}
                    name="business_name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Business Name</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. Beauty Salon AI" {...field} />
                        </FormControl>
                        <FormDescription>
                          The name the AI will use to refer to itself and your business.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="tone"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Communication Tone</FormLabel>
                        <Select onValueChange={field.onChange} value={field.value || "professional"}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select a tone" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="professional">Professional (Courteous & Formal)</SelectItem>
                            <SelectItem value="friendly">Friendly (Warm & Approachable)</SelectItem>
                            <SelectItem value="casual">Casual (Relaxed & Conversational)</SelectItem>
                            <SelectItem value="formal">Formal (Strictly Business)</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormDescription>
                          Sets the overall style of the AI's responses.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="behavior_rules"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Behavior Rules</FormLabel>
                        <FormControl>
                          <Textarea 
                            placeholder="- Always ask for a name before booking&#10;- Do not mention competitor prices&#10;- Speak only in Arabic" 
                            className="min-h-[120px]"
                            {...field} 
                          />
                        </FormControl>
                        <FormDescription>
                          Specific guidelines the AI must follow (one per line).
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="advanced" className="space-y-4 mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Advanced Configuration</CardTitle>
                  <CardDescription>Fine-tune the AI's behavior with custom instructions.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Warning</AlertTitle>
                    <AlertDescription>
                      Custom instructions override the basic settings (Tone, Rules). Use this only if you need complete control over the system prompt.
                    </AlertDescription>
                  </Alert>

                  <FormField
                    control={form.control}
                    name="custom_instructions"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>System Prompt Override</FormLabel>
                        <FormControl>
                          <Textarea 
                            placeholder="You are a helpful assistant..." 
                            className="min-h-[300px] font-mono text-sm"
                            {...field} 
                          />
                        </FormControl>
                        <FormDescription>
                          The raw system prompt sent to the LLM. Leave empty to use the generated prompt from Basic Settings.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="ai_pause_duration_minutes"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Default Pause Duration (Minutes)</FormLabel>
                        <FormControl>
                          <Input type="number" {...field} />
                        </FormControl>
                        <FormDescription>
                          How long the AI should sleep when you click "Pause" or intervene manually.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="flex justify-end">
            <Button type="submit" disabled={updateConfigMutation.isPending}>
              {updateConfigMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              <Save className="mr-2 h-4 w-4" />
              Save Configuration
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
