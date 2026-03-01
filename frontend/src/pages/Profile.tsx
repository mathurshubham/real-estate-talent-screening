import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Label } from "@/components/ui/label"
import { Lock } from "lucide-react"

export function Profile() {
    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div>
                <h3 className="text-2xl font-bold tracking-tight">Profile Settings</h3>
                <p className="text-muted-foreground">
                    Manage your account settings and preferences.
                </p>
            </div>

            <Tabs defaultValue="account" className="w-full">
                <TabsList className="mb-4">
                    <TabsTrigger value="account">Account Info</TabsTrigger>
                    <TabsTrigger value="security">Security</TabsTrigger>
                </TabsList>
                <TabsContent value="account">
                    <div className="border rounded-lg p-6 space-y-8 bg-card text-card-foreground shadow-sm">
                        <div className="space-y-4">
                            <div>
                                <h4 className="text-sm font-medium mb-4">Personal Information</h4>
                            </div>
                            <div className="flex items-center gap-6">
                                <Avatar className="h-20 w-20">
                                    <AvatarImage src="" />
                                    <AvatarFallback className="text-2xl font-semibold">SG</AvatarFallback>
                                </Avatar>
                                <Button variant="outline" size="sm">Change photo</Button>
                            </div>
                        </div>

                        <div className="grid gap-4 md:grid-cols-2">
                            <div className="space-y-2">
                                <Label htmlFor="name">Full Name</Label>
                                <Input id="name" defaultValue="Sudhir Gupta" />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="email">Email Address</Label>
                                <div className="flex gap-2">
                                    <Input id="email" defaultValue="sudhir@estateassess.com" disabled />
                                    <Button variant="secondary" disabled>Verified</Button>
                                </div>
                            </div>
                        </div>

                        <Button>Save Changes</Button>
                    </div>
                </TabsContent>
                <TabsContent value="security">
                    <div className="border rounded-lg p-6 space-y-8 bg-card text-card-foreground shadow-sm">
                        <div>
                            <h4 className="text-sm font-medium mb-4">Change Password</h4>
                        </div>

                        <div className="max-w-md space-y-4">
                            <div className="space-y-2 relative">
                                <Label htmlFor="current-password">Current Password</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                                    <Input id="current-password" type="password" placeholder="••••••••" className="pl-9" />
                                </div>
                            </div>
                            <div className="space-y-2 relative">
                                <Label htmlFor="new-password">New Password</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                                    <Input id="new-password" type="password" placeholder="••••••••" className="pl-9" />
                                </div>
                            </div>
                            <div className="space-y-2 relative">
                                <Label htmlFor="confirm-password">Confirm Password</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                                    <Input id="confirm-password" type="password" placeholder="••••••••" className="pl-9" />
                                </div>
                            </div>
                        </div>

                        <Button>Update Password</Button>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    )
}
