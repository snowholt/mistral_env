import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Link } from "react-router-dom";
import { Building2, Loader2, Plus, ArrowRight } from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useAuth } from "@/hooks/useAuth";
import { useCustomers, useCreateCustomer } from "@/hooks/useAgent";

const createBusinessSchema = z.object({
  name: z.string().min(2, "Business name must be at least 2 characters").max(255),
  email: z.string().email("Please enter a valid email").max(255),
});

export default function Businesses() {
  const { user } = useAuth();
  const { data: customers, isLoading } = useCustomers();
  const createCustomerMutation = useCreateCustomer();

  const form = useForm<z.infer<typeof createBusinessSchema>>({
    resolver: zodResolver(createBusinessSchema),
    defaultValues: {
      name: "",
      email: user?.email || "",
    },
  });

  useEffect(() => {
    if (user?.email) {
      form.setValue("email", user.email);
    }
  }, [user?.email, form]);

  const onSubmit = (values: z.infer<typeof createBusinessSchema>) => {
    createCustomerMutation.mutate(
      { name: values.name, email: values.email },
      {
        onSuccess: (created) => {
          toast.success("Business created successfully");
          form.reset({ name: "", email: created.email });
        },
        onError: (error: any) => {
          toast.error(error?.detail || "Failed to create business");
        },
      }
    );
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Building2 className="h-6 w-6" />
          Businesses
        </h1>
        <p className="text-gray-600 mt-1">Create and manage your business profile.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Create Business</CardTitle>
          <CardDescription>
            This creates a business profile in the backend ("customer") so you can configure the AI agent.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Business Name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g. Kesay Clinics" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Business Email</FormLabel>
                    <FormControl>
                      <Input placeholder="business@email.com" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="flex justify-end">
                <Button type="submit" disabled={createCustomerMutation.isPending}>
                  {createCustomerMutation.isPending ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Plus className="mr-2 h-4 w-4" />
                  )}
                  Create Business
                </Button>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Your Businesses</CardTitle>
          <CardDescription>Businesses linked to your account.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-10">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          ) : !customers || customers.length === 0 ? (
            <div className="text-sm text-gray-600">No businesses yet. Create one above.</div>
          ) : (
            <div className="space-y-3">
              {customers.map((c) => (
                <div key={c.id} className="flex items-center justify-between rounded-lg border p-3">
                  <div>
                    <div className="font-medium text-gray-900">{c.name}</div>
                    <div className="text-sm text-gray-600">{c.email}</div>
                  </div>
                  <Link to="/app/agent">
                    <Button variant="outline" size="sm" className="gap-2">
                      Configure Agent
                      <ArrowRight className="h-4 w-4" />
                    </Button>
                  </Link>
                </div>
              ))}
            </div>
          )}
        </CardContent>
        <CardFooter className="justify-end">
          <Link to="/app/agent">
            <Button variant="ghost" className="gap-2">
              Go to AI Agent
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </CardFooter>
      </Card>
    </div>
  );
}
