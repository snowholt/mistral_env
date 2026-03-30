import { useEffect, useState } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { authApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

type VerificationState = "loading" | "success" | "error";

export default function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [state, setState] = useState<VerificationState>("loading");
  const [message, setMessage] = useState("");

  const token = searchParams.get("token");

  useEffect(() => {
    const verify = async () => {
      if (!token) {
        setState("error");
        setMessage("رابط التحقق غير صالح. الرجاء طلب رابط جديد.");
        return;
      }

      try {
        const response = await authApi.verifyEmail(token);
        setState("success");
        setMessage(response.message || "تم تأكيد بريدك الإلكتروني بنجاح!");
        
        // Redirect to login after 3 seconds
        setTimeout(() => {
          navigate("/login", { 
            state: { message: "تم تأكيد بريدك الإلكتروني. يمكنك الآن تسجيل الدخول." }
          });
        }, 3000);
      } catch (error: any) {
        setState("error");
        setMessage(error.message || "فشل التحقق. الرابط قد يكون منتهي الصلاحية أو غير صالح.");
      }
    };

    verify();
  }, [token, navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted p-4" dir="rtl">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">
            تأكيد البريد الإلكتروني
          </CardTitle>
          <CardDescription>
            Email Verification
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-6">
          {state === "loading" && (
            <>
              <Loader2 className="h-16 w-16 text-primary animate-spin" />
              <p className="text-muted-foreground text-center">
                جارٍ التحقق من بريدك الإلكتروني...
                <br />
                <span className="text-sm">Verifying your email...</span>
              </p>
            </>
          )}

          {state === "success" && (
            <>
              <CheckCircle className="h-16 w-16 text-green-500" />
              <div className="text-center">
                <p className="text-lg font-medium text-green-600 mb-2">
                  {message}
                </p>
                <p className="text-muted-foreground text-sm">
                  سيتم توجيهك إلى صفحة تسجيل الدخول...
                  <br />
                  Redirecting to login...
                </p>
              </div>
              <Button asChild className="w-full">
                <Link to="/login">تسجيل الدخول | Login</Link>
              </Button>
            </>
          )}

          {state === "error" && (
            <>
              <XCircle className="h-16 w-16 text-red-500" />
              <div className="text-center">
                <p className="text-lg font-medium text-red-600 mb-2">
                  فشل التحقق
                </p>
                <p className="text-muted-foreground">
                  {message}
                </p>
              </div>
              <div className="flex flex-col gap-2 w-full">
                <Button asChild variant="outline" className="w-full">
                  <Link to="/login">العودة لتسجيل الدخول | Back to Login</Link>
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
