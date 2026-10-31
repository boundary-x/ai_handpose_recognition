(function(root) {
  class HandEngineRecovery {
    constructor({create, onState, timeout = 45000}) {
      Object.assign(this, {create, onState, timeout, engine:null, mode:'GPU', state:'idle', epoch:0, errors:0, autoUsed:false});
    }
    async load(mode = 'GPU', automatic = false) {
      if (this.state === 'loading') return false;
      const epoch = ++this.epoch;
      this.state = 'loading'; this.errors = 0; this.mode = mode;
      if (automatic || mode === 'CPU') this.autoUsed = true;
      try { this.engine?.close(); } catch (_) {}
      this.engine = null;
      this.onState('loading', mode);
      let timer, expired = false;
      try {
        const creation = Promise.resolve().then(()=>this.create(mode)).then(engine=>{
          if (expired || epoch !== this.epoch) { engine.close(); throw Error('Expired model load'); }
          return engine;
        });
        const deadline = new Promise((_,reject)=>{timer=setTimeout(()=>{expired=true;reject(Error('모델 준비 시간이 초과되었습니다. 네트워크 연결을 확인해주세요.'));},this.timeout);});
        this.engine = await Promise.race([creation, deadline]);
        this.state = 'ready'; this.onState('ready', mode); return true;
      } catch(error) {
        this.state = 'error';
        if (mode === 'GPU' && !this.autoUsed) return this.load('CPU', true);
        this.onState('error', mode, error); return false;
      } finally { clearTimeout(timer); }
    }
    detect(input, now) {
      if (this.state !== 'ready') return null;
      try {
        const result = this.engine.detectForVideo(input, now);
        this.errors = 0; return result;
      } catch(error) {
        this.errors++;
        this.onState('inference-error', this.mode, error);
        if (this.errors >= 3) {
          if (this.mode === 'GPU' && !this.autoUsed) void this.load('CPU', true);
          else { this.state = 'error'; this.onState('error', this.mode, error); }
        }
        return null;
      }
    }
  }
  root.HandEngineRecovery = HandEngineRecovery;
  if (typeof module !== 'undefined') module.exports = HandEngineRecovery;
})(globalThis);

