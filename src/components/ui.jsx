import React from 'react';
import { cn } from '../lib/utils';

// Button
export const Button = React.forwardRef(({ className, variant = 'primary', size = 'default', ...props }, ref) => {
    const variants = {
        primary: "bg-[#1A365D] text-white hover:bg-[#2A4365] shadow-md",
        accent: "bg-[#D4AF37] text-white hover:bg-[#B8860B] shadow-md",
        outline: "border border-[#1A365D] text-[#1A365D] hover:bg-slate-50",
        ghost: "hover:bg-slate-100 text-slate-600",
    };
    const sizes = {
        default: "h-10 px-4 py-2",
        sm: "h-9 px-3",
        lg: "h-11 px-8",
        icon: "h-10 w-10",
    };
    return (
        <button
            className={cn(
                "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
                variants[variant],
                sizes[size],
                className
            )}
            ref={ref}
            {...props}
        />
    );
});

// Card
export const Card = ({ className, ...props }) => (
    <div className={cn("rounded-xl border bg-card text-card-foreground shadow", className)} {...props} />
);

export const CardHeader = ({ className, ...props }) => (
    <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
);

export const CardTitle = ({ className, ...props }) => (
    <h3 className={cn("font-semibold leading-none tracking-tight text-xl text-[#1A365D]", className)} {...props} />
);

export const CardContent = ({ className, ...props }) => (
    <div className={cn("p-6 pt-0", className)} {...props} />
);

// Input
export const Input = React.forwardRef(({ className, type, ...props }, ref) => (
    <input
        type={type}
        className={cn(
            "flex h-10 w-100 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
            className
        )}
        ref={ref}
        {...props}
    />
));

// Modal (Simple for this project)
export const Modal = ({ isOpen, onClose, title, children }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="w-full max-w-md bg-white rounded-xl shadow-2xl animate-in zoom-in-95 duration-200">
                <CardHeader className="flex flex-row items-center justify-between border-b">
                    <CardTitle>{title}</CardTitle>
                    <button onClick={onClose} className="text-slate-400 hover:text-slate-600">×</button>
                </CardHeader>
                <div className="p-6">{children}</div>
            </div>
        </div>
    );
};
