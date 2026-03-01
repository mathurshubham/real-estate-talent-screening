import { useState } from "react"
import { Link } from "react-router-dom"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
    LayoutDashboard,
    Users,
    FileText,
    ShieldAlert,
    Settings,
    Building,
    ChevronLeft,
    ChevronRight
} from "lucide-react"

interface SidebarProps {
    className?: string
}

export function Sidebar({ className }: SidebarProps) {
    const [isCollapsed, setIsCollapsed] = useState(false)

    const navItems = [
        { title: "Dashboard", icon: LayoutDashboard, href: "/dashboard" },
        { title: "Candidates", icon: Users, href: "/candidates" },
        { title: "Assessments", icon: FileText, href: "/assessments" },
        { title: "Users", icon: ShieldAlert, href: "/admin", adminOnly: true },
        { title: "Organization", icon: Building, href: "/organization" },
    ]

    return (
        <div
            className={cn(
                "relative hidden h-[calc(100vh-3.5rem)] border-r bg-background md:block transition-all duration-300",
                isCollapsed ? "w-16" : "w-64",
                className
            )}
        >
            <div className="flex h-full flex-col justify-between py-4">
                <div className="px-3 py-2">
                    <div className="space-y-1">
                        {navItems.map((item) => (
                            <Button
                                key={item.href}
                                variant="ghost"
                                className={cn(
                                    "w-full justify-start",
                                    isCollapsed ? "px-2" : "px-4"
                                )}
                                asChild
                            >
                                <Link to={item.href}>
                                    <item.icon className={cn("h-5 w-5", isCollapsed ? "mr-0" : "mr-2")} />
                                    {!isCollapsed && <span>{item.title}</span>}
                                </Link>
                            </Button>
                        ))}
                    </div>
                </div>

                <div className="px-3 py-2 space-y-2">
                    <Button
                        variant="ghost"
                        className={cn(
                            "w-full justify-start",
                            isCollapsed ? "px-2" : "px-4"
                        )}
                        asChild
                    >
                        <Link to="/profile">
                            <Settings className={cn("h-5 w-5", isCollapsed ? "mr-0" : "mr-2")} />
                            {!isCollapsed && <span>Settings</span>}
                        </Link>
                    </Button>

                    <div className="flex justify-end px-2 pt-2 border-t">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setIsCollapsed(!isCollapsed)}
                            className="h-6 w-6"
                        >
                            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    )
}
