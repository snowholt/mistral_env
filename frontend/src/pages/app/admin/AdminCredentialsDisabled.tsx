import { Link } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { KeyRound, ArrowRight } from "lucide-react";

export default function AdminCredentialsDisabled() {
  return (
    <div className="flex items-center justify-center min-h-[60vh] px-4">
      <Card className="max-w-xl w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <KeyRound className="h-5 w-5" />
            Credentials Disabled
          </CardTitle>
          <CardDescription>
            Admin credential management is disabled for security.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Please update or add tokens from the WhatsApp settings page under the Token section.
          </p>
          <Button asChild>
            <Link to="/app/whatsapp">
              Go to WhatsApp Settings
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
