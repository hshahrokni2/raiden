#!/bin/bash
# Raiden API - Fly.io Deployment Script
# 
# Prerequisites:
#   1. Install flyctl: curl -L https://fly.io/install.sh | sh
#   2. Login: fly auth login
#
# Usage:
#   ./deploy.sh              # First-time setup + deploy
#   ./deploy.sh --deploy     # Deploy only (after initial setup)

set -e

echo "🚀 Raiden API Deployment"
echo "========================"

# Check if flyctl is installed
if ! command -v fly &> /dev/null; then
    echo "❌ flyctl not found. Install with: curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if logged in
if ! fly auth whoami &> /dev/null; then
    echo "❌ Not logged in. Run: fly auth login"
    exit 1
fi

echo "✅ Fly.io CLI ready"

# Check for required env vars locally (for reference)
if [ -f .env ]; then
    echo "📄 Found .env file"
    source .env
fi

if [ "$1" == "--deploy" ]; then
    # Deploy only
    echo "📦 Deploying to Fly.io..."
    fly deploy
else
    # First-time setup
    echo "🔧 Setting up Fly.io app..."
    
    # Launch app (creates fly.toml if not exists, creates app on Fly)
    fly launch --no-deploy --copy-config --name raiden-api --region arn
    
    echo ""
    echo "🔐 Setting secrets..."
    echo "   You'll need to provide:"
    echo "   - KOMILION_API_KEY (for LLM calls)"
    echo "   - GOOGLE_API_KEY (for Street View)"
    echo ""
    
    # Prompt for secrets
    read -p "KOMILION_API_KEY: " KOMILION_KEY
    read -p "GOOGLE_API_KEY: " GOOGLE_KEY
    
    fly secrets set \
        KOMILION_API_KEY="$KOMILION_KEY" \
        GOOGLE_API_KEY="$GOOGLE_KEY"
    
    echo ""
    echo "📦 Deploying..."
    fly deploy
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 API URL: https://raiden-api.fly.dev"
echo ""
echo "Test with:"
echo "  curl https://raiden-api.fly.dev/health"
echo ""
echo "Visual analysis:"
echo "  curl -X POST https://raiden-api.fly.dev/visual/analyze \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"lat\": 59.3044309, \"lon\": 18.0937078}'"

