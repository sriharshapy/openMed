cat > ~/.ngrok2/ngrok.yml << 'EOF'
version: "2"
authtoken: 2dAwoP0pcszmx6zDX7uQNh6ArsG_2Wz64sPR9cNRXa6aZN2RT

tunnels:
  backend:
    addr: 8000
    proto: http
    
  frontend:
    addr: 3000
    proto: http
    
  extra:
    addr: 6010
    proto: http
EOF

# Step 3: Replace YOUR_TOKEN_HERE with your actual token
# If you have token in a file:
if [ -f "ngrok_token.txt" ]; then
    TOKEN=$(cat ngrok_token.txt | tr -d '\n\r ')
    sed -i "s/YOUR_TOKEN_HERE/$TOKEN/" ~/.ngrok2/ngrok.yml
    echo "✅ Token configured from ngrok_token.txt"
elif [ -f "token.txt" ]; then
    TOKEN=$(cat token.txt | tr -d '\n\r ')
    sed -i "s/YOUR_TOKEN_HERE/$TOKEN/" ~/.ngrok2/ngrok.yml
    echo "✅ Token configured from token.txt"
else
    echo "⚠️  Please manually edit ~/.ngrok2/ngrok.yml and replace YOUR_TOKEN_HERE with your actual token"
fi