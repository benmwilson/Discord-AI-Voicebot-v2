// Shared Navbar JavaScript for Miku Bot Dashboard
// This file provides common navbar functionality across all pages

// Global navbar state management
window.NavbarManager = {
    botStatus: {
        online: false,
        guilds: [],
        users: 0,
        latency: 0,
        uptime: '0:00:00',
        voice_channels: [],
        in_voice: false
    },
    
    // Initialize navbar on any page
    init() {
        console.log('Initializing shared navbar...');
        this.showLoadingPlaceholder();
        this.loadNavbar();
        this.initWebSocket();
        this.updateActiveLinks();
    },
    
    // Show loading placeholder to prevent flash
    showLoadingPlaceholder() {
        const navbarContainer = document.getElementById('shared-navbar');
        if (navbarContainer) {
            navbarContainer.classList.add('loading');
            navbarContainer.innerHTML = `
                <nav class="bg-white shadow-sm border-b border-gray-200" aria-label="Miku Bot Dashboard menu">
                    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div class="flex justify-between items-center py-4">
                            <!-- Brand Logo -->
                            <a href="/" class="flex items-center space-x-3">
                                <div class="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                                    <span class="text-white text-xl">🤖</span>
                                </div>
                                <div>
                                    <h1 class="text-2xl font-bold text-gray-900">Miku Bot</h1>
                                    <p class="text-sm text-gray-500">Dashboard</p>
                                </div>
                            </a>
                            
                            <!-- Desktop Menu -->
                            <ul class="hidden md:flex items-center space-x-8">
                                <li><a href="/" class="text-gray-700 hover:text-purple-600 transition-colors font-medium">Dashboard</a></li>
                                <li><a href="/enhanced" class="text-gray-700 hover:text-purple-600 transition-colors font-medium">🎨 Enhanced</a></li>
                                <li><a href="/memory" class="text-gray-700 hover:text-purple-600 transition-colors font-medium">🧠 Memory</a></li>
                                <li><a href="#" class="text-gray-700 hover:text-purple-600 transition-colors font-medium">📊 Analytics</a></li>
                                <li><a href="#" class="text-gray-700 hover:text-purple-600 transition-colors font-medium">⚙️ Settings</a></li>
                                
                                <!-- Status Indicator -->
                                <li class="flex items-center space-x-2">
                                    <div class="w-3 h-3 rounded-full bg-gray-400 animate-pulse" id="statusIndicator"></div>
                                    <span class="text-sm font-medium text-gray-500" id="statusText">Loading...</span>
                                </li>
                                
                                <!-- Last Updated -->
                                <li class="text-sm text-gray-500" id="lastUpdated">Loading...</li>
                            </ul>
                            
                            <!-- Mobile Menu Button -->
                            <button class="md:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors" aria-label="mobile menu">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" aria-hidden="true" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </nav>
            `;
        }
    },
    
    // Load the actual navbar
    async loadNavbar() {
        try {
            const response = await fetch('/static/shared_navbar.html');
            const navbarHtml = await response.text();
            const navbarContainer = document.getElementById('shared-navbar');
            if (navbarContainer) {
                navbarContainer.innerHTML = navbarHtml;
                navbarContainer.classList.remove('loading');
                console.log('Shared navbar loaded successfully');
                // Re-initialize Alpine.js for the new content
                if (window.Alpine) {
                    window.Alpine.initTree(navbarContainer);
                }
            }
        } catch (error) {
            console.error('Error loading shared navbar:', error);
            // Keep the placeholder if loading fails
            const navbarContainer = document.getElementById('shared-navbar');
            if (navbarContainer) {
                navbarContainer.classList.remove('loading');
            }
        }
    },
    
    // Initialize WebSocket connection for real-time updates
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        ws.onopen = function(event) {
            console.log('Navbar WebSocket connected');
        };
        
        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                console.log('Navbar received update:', data);
                NavbarManager.updateStatus(data);
            } catch (e) {
                console.error('Error parsing navbar WebSocket message:', e);
            }
        };
        
        ws.onclose = function(event) {
            console.log('Navbar WebSocket disconnected');
            // Set offline status
            NavbarManager.updateStatus({ online: false });
            // Reconnect after 3 seconds
            setTimeout(() => NavbarManager.initWebSocket(), 3000);
        };
        
        ws.onerror = function(error) {
            console.log('Navbar WebSocket error:', error);
        };
    },
    
    // Update navbar status indicators
    updateStatus(data) {
        // Update global status
        this.botStatus = { ...this.botStatus, ...data };
        
        // Update desktop status indicators
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const statusIndicatorMobile = document.getElementById('statusIndicatorMobile');
        const statusTextMobile = document.getElementById('statusTextMobile');
        
        if (statusIndicator) {
            statusIndicator.className = `w-3 h-3 rounded-full ${data.online ? 'bg-green-500' : 'bg-red-500'}`;
        }
        if (statusText) {
            statusText.textContent = data.online ? 'Online' : 'Offline';
        }
        if (statusIndicatorMobile) {
            statusIndicatorMobile.className = `w-3 h-3 rounded-full ${data.online ? 'bg-green-500' : 'bg-red-500'}`;
        }
        if (statusTextMobile) {
            statusTextMobile.textContent = data.online ? 'Online' : 'Offline';
        }
        
        // Update last updated time (desktop and mobile)
        const lastUpdated = document.getElementById('lastUpdated');
        const lastUpdatedMobile = document.getElementById('lastUpdatedMobile');
        const timeString = `Last updated: ${new Date().toLocaleTimeString()}`;
        
        if (lastUpdated) {
            lastUpdated.textContent = timeString;
        }
        if (lastUpdatedMobile) {
            lastUpdatedMobile.textContent = timeString;
        }
        
        console.log('Navbar status updated:', data.online ? 'Online' : 'Offline');
    },
    
    // Update active navigation links
    updateActiveLinks() {
        const currentPath = window.location.pathname;
        const links = document.querySelectorAll('nav a[href]');
        
        links.forEach(link => {
            const linkPath = new URL(link.href).pathname;
            if (linkPath === currentPath) {
                link.classList.add('text-purple-600');
                link.classList.remove('text-gray-700');
            } else {
                link.classList.remove('text-purple-600');
                link.classList.add('text-gray-700');
            }
        });
    }
};

// Add CSS to prevent flash
const style = document.createElement('style');
style.textContent = `
    #shared-navbar {
        opacity: 1;
        transition: opacity 0.2s ease-in-out;
    }
    #shared-navbar.loading {
        opacity: 0.8;
    }
`;
document.head.appendChild(style);

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (window.NavbarManager) {
        window.NavbarManager.init();
    }
});
