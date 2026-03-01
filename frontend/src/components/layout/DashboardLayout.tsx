import { useState } from "react"
import { Link } from "react-router-dom"
import { TopNav } from "./TopNav"
import { Sidebar } from "./Sidebar"
import { Footer } from "./Footer"
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet"
import { ThemeProvider } from "@/components/theme-provider"
import { LayoutDashboard, Users, FileText, ShieldAlert, Building, Settings } from "lucide-react"

interface DashboardLayoutProps {
    children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

    const navItems = [
        { title: "Dashboard", icon: LayoutDashboard, href: "/dashboard" },
        { title: "Candidates", icon: Users, href: "/candidates" },
        { title: "Assessments", icon: FileText, href: "/assessments" },
        { title: "Users", icon: ShieldAlert, href: "/admin" },
        { title: "Organization", icon: Building, href: "/organization" },
        { title: "Settings", icon: Settings, href: "/profile" },
    ]

    return (
        <ThemeProvider defaultTheme="system" storageKey="estateassess-theme">
            <div className="relative flex min-h-screen flex-col bg-background">
                <TopNav onMenuClick={() => setMobileMenuOpen(true)} />

                <div className="flex flex-1 overflow-hidden">
                    <Sidebar />

                    <main className="flex-1 overflow-y-auto w-full">
                        <div className="container p-4 md:p-6 lg:p-8 mx-auto max-w-7xl">
                            {children}
                        </div>
                        <Footer />
                    </main>
                </div>

                {/* Mobile Sidebar */}
                <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
                    <SheetContent side="left" className="w-[280px] sm:w-[320px] p-0">
                        <SheetTitle className="sr-only">Mobile Navigation Menu</SheetTitle>
                        <div className="flex h-full flex-col py-6">
                            <div className="px-6 pb-4 border-b">
                                <span className="font-bold text-lg">EstateAssess</span>
                            </div>
                            <div className="flex-1 overflow-auto py-4">
                                <nav className="grid gap-2 px-4">
                                    {navItems.map((item) => (
                                        <Link
                                            key={item.href}
                                            to={item.href}
                                            onClick={() => setMobileMenuOpen(false)}
                                            className="flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground"
                                        >
                                            <item.icon className="h-5 w-5" />
                                            {item.title}
                                        </Link>
                                    ))}
                                </nav>
                            </div>
                        </div>
                    </SheetContent>
                </Sheet>
            </div>
        </ThemeProvider>
    )
}
