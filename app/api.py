import os, json, shutil, uuid, datetime as dt
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from app.pipeline import assess

DB = os.getenv("DATABASE_URL", "sqlite:///./claims.db")
engine = create_engine(DB); Session = sessionmaker(engine); Base = declarative_base()

class Claim(Base):
    __tablename__ = "claims"
    id = Column(String, primary_key=True); created = Column(DateTime, default=dt.datetime.utcnow)
    description = Column(Text); result = Column(Text); files = Column(Text)
    status = Column(String, default="PENDING_REVIEW"); reviewer_note = Column(Text, default="")
Base.metadata.create_all(engine)

app = FastAPI(title="Claim Assessment API")
def db():
    s = Session()
    try: yield s
    finally: s.close()

@app.post("/claims")
def create_claim(description: str = Form(""), vehicle_age: int = Form(5),
                 images: list[UploadFile] = File(...), documents: list[UploadFile] = File(default=[]), s=Depends(db)):
    cid = f"CLM-{dt.date.today().year}-{uuid.uuid4().hex[:6].upper()}"
    folder = f"uploads/{cid}"; os.makedirs(folder, exist_ok=True)
    def save(fs):
        out = []
        for f in fs:
            p = f"{folder}/{os.path.basename(f.filename)}"
            with open(p, "wb") as o: shutil.copyfileobj(f.file, o)
            out.append(p)
        return out
    ip, dp = save(images), save(documents)
    result = assess(ip, dp, description, vehicle_age)
    s.add(Claim(id=cid, description=description, result=json.dumps(result), files=json.dumps(ip + dp))); s.commit()
    return {"claim_id": cid, "status": "PENDING_REVIEW", **result}

@app.get("/claims")
def list_claims(s=Depends(db)):
    return [{"id": c.id, "created": c.created.isoformat(), "status": c.status} for c in s.query(Claim).order_by(Claim.created.desc())]

@app.get("/claims/{cid}")
def get_claim(cid: str, s=Depends(db)):
    c = s.get(Claim, cid)
    if not c: raise HTTPException(404)
    return {"id": c.id, "status": c.status, "description": c.description, "files": json.loads(c.files),
            "result": json.loads(c.result), "reviewer_note": c.reviewer_note}

@app.patch("/claims/{cid}/decision")
def decide(cid: str, status: str = Form(...), note: str = Form(""), s=Depends(db)):
    if status not in {"APPROVED", "REJECTED", "NEEDS_INFO"}: raise HTTPException(400, "bad status")
    c = s.get(Claim, cid)
    if not c: raise HTTPException(404)
    c.status, c.reviewer_note = status, note; s.commit()
    return {"id": cid, "status": status}
