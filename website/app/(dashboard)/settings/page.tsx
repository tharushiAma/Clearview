"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { useSettings } from "@/lib/settings-context";
import { Save, RotateCcw } from "lucide-react";
import { useState } from "react";

export default function SettingsPage() {
  const { settings, updateSettings } = useSettings();
  const [localSettings, setLocalSettings] = useState(settings);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    updateSettings(localSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    const defaults = {
      apiBaseUrl: "http://localhost:8000",
      defaultCheckpointPath: "/models/clearview-absa-v1",
      darkMode: false,
    };
    setLocalSettings(defaults);
    updateSettings(defaults);
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Configure the ClearView dashboard
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">API Configuration</CardTitle>
          <CardDescription>
            Configure the backend API connection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="api-url">API Base URL</Label>
            <Input
              id="api-url"
              value={localSettings.apiBaseUrl}
              onChange={(e) =>
                setLocalSettings((s) => ({ ...s, apiBaseUrl: e.target.value }))
              }
              placeholder="http://localhost:8000"
            />
            <p className="text-xs text-muted-foreground">
              The base URL for the ClearView API server
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="checkpoint">Default Checkpoint Path</Label>
            <Input
              id="checkpoint"
              value={localSettings.defaultCheckpointPath}
              onChange={(e) =>
                setLocalSettings((s) => ({
                  ...s,
                  defaultCheckpointPath: e.target.value,
                }))
              }
              placeholder="/models/clearview-absa-v1"
            />
            <p className="text-xs text-muted-foreground">
              Path to the default model checkpoint
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Appearance</CardTitle>
          <CardDescription>
            Customize the dashboard appearance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="dark-mode">Dark Mode</Label>
              <p className="text-xs text-muted-foreground">
                Toggle dark mode for the dashboard
              </p>
            </div>
            <Switch
              id="dark-mode"
              checked={localSettings.darkMode}
              onCheckedChange={(checked) => {
                setLocalSettings((s) => ({ ...s, darkMode: checked }));
                updateSettings({ darkMode: checked });
              }}
            />
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center gap-3">
        <Button onClick={handleSave}>
          <Save className="h-4 w-4 mr-2" />
          {saved ? "Saved!" : "Save Settings"}
        </Button>
        <Button variant="outline" onClick={handleReset}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset to Defaults
        </Button>
      </div>
    </div>
  );
}
