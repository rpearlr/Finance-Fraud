import hashlib
import os
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Response, Request, Header, Cookie
from backend.database import query_one, execute_db
from backend.schemas import UserRegister, UserLogin, AuthResponse
from backend.config import log, JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(tags=["Auth"])

def hash_password(password: str, salt: str = None):
    """Securely hash a password using PBKDF2."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Standard PBKDF2 HMAC-SHA256
    dk = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    )
    return dk.hex(), salt

def verify_password(password: str, salt: str, hashed: str):
    """Verify a password against its hash and salt."""
    new_hash, _ = hash_password(password, salt)
    return new_hash == hashed

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_current_user(
    request: Request, 
    authorization: Optional[str] = Header(None), 
    session_token: Optional[str] = Cookie(None)
):
    """Dependency to verify the JWT and return the user details."""
    token = None
    
    # 1. Try Authorization Header (Bearer token)
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    
    # 2. Fallback to session_token Cookie
    if not token and session_token:
        token = session_token

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Decode and verify the JWT
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if username is None or user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
            
        return {"username": username, "user_id": user_id}
        
    except jwt.ExpiredSignatureError:
        log.warning("JWT token expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError as e:
        log.warning(f"JWT validation error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@router.post("/auth/register", response_model=AuthResponse)
def register(req: UserRegister):
    # Check if user exists
    existing = query_one("SELECT * FROM users WHERE username = ? OR email = ?", (req.username, req.email))
    if existing:
        raise HTTPException(status_code=400, detail="Username or Email already registered")
    
    hashed, salt = hash_password(req.password)
    
    try:
        user_id = execute_db(
            "INSERT INTO users (username, email, password, salt) VALUES (?, ?, ?, ?)",
            (req.username, req.email, hashed, salt)
        )
        log.info(f"New user registered: {req.username}")
        return AuthResponse(
            status="ok",
            message="User registered successfully",
            user_id=user_id,
            username=req.username
        )
    except Exception as e:
        log.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during registration")

@router.post("/auth/login", response_model=AuthResponse)
def login(req: UserLogin, response: Response):
    # Only check username as requested
    user = query_one("SELECT * FROM users WHERE username = ?", (req.username,))
    
    if not user or not verify_password(req.password, user["salt"], user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]}, 
        expires_delta=access_token_expires
    )
    
    log.info(f"User logged in: {req.username}")
    
    # Optionally set a cookie for convenience
    response.set_cookie(key="session_token", value=token, httponly=True)
    
    return AuthResponse(
        status="ok",
        message="Login successful",
        user_id=user["id"],
        username=user["username"],
        token=token
    )

@router.post("/auth/logout", response_model=AuthResponse)
def logout(response: Response):
    response.delete_cookie(key="session_token")
    return AuthResponse(
        status="ok",
        message="Logged out successfully"
    )
