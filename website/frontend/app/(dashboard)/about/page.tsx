import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Eye, Sparkles, Activity, Layers } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">About</h1>
        <p className="text-muted-foreground">
          Information about the ClearView project
        </p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Eye className="h-5 w-5" />
            </div>
            <div>
              <CardTitle>ClearView</CardTitle>
              <CardDescription>
                Cosmetics ABSA + MSR + XAI Dashboard
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            ClearView is a comprehensive dashboard for Aspect-Based Sentiment Analysis (ABSA)
            of cosmetics reviews. It incorporates Mixed Sentiment Resolution (MSR) to handle
            conflicting sentiments and provides Explainable AI (XAI) capabilities for model interpretability.
          </p>

          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">ABSA</Badge>
            <Badge variant="secondary">Mixed Sentiment Resolution</Badge>
            <Badge variant="secondary">Explainable AI</Badge>
            <Badge variant="secondary">Cosmetics Domain</Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        <FeatureCard
          icon={Layers}
          title="ABSA"
          description="Analyze sentiment across 7 cosmetics-specific aspects including texture, smell, price, and more."
        />
        <FeatureCard
          icon={Activity}
          title="MSR"
          description="Mixed Sentiment Resolution handles conflicting sentiments in complex reviews."
        />
        <FeatureCard
          icon={Sparkles}
          title="XAI"
          description="Explainable AI with Integrated Gradients, LIME, and SHAP for model transparency."
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Aspects Analyzed</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { name: "Staying Power", desc: "How long it lasts" },
              { name: "Texture", desc: "Feel and consistency" },
              { name: "Smell", desc: "Fragrance quality" },
              { name: "Price", desc: "Value for money" },
              { name: "Colour", desc: "Color accuracy" },
              { name: "Shipping", desc: "Delivery experience" },
              { name: "Packing", desc: "Packaging quality" },
            ].map((aspect) => (
              <div key={aspect.name} className="p-3 rounded-lg bg-muted/50">
                <p className="text-sm font-medium">{aspect.name}</p>
                <p className="text-xs text-muted-foreground">{aspect.desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Separator />

      <div className="text-sm text-muted-foreground">
        <p className="mt-1">
          Designed for thesis demonstration purposes.
        </p>
      </div>
    </div>
  );
}

function FeatureCard({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <Icon className="h-8 w-8 mb-3 text-muted-foreground" />
        <h3 className="font-medium">{title}</h3>
        <p className="text-sm text-muted-foreground mt-1">{description}</p>
      </CardContent>
    </Card>
  );
}
