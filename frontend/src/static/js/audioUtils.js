// Shared audio utility functions (centralized)
// Provides Float32 -> Int16 conversion used by debug pages.
// Keeping implementation minimal and synchronous.

(function(global){
  function floatToInt16(float32){
    const out = new Int16Array(float32.length);
    for(let i=0;i<float32.length;i++){
      let s = float32[i];
      if (s > 1) s = 1; else if (s < -1) s = -1;
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
  }

  function floatToInt16Buffer(float32){
    return floatToInt16(float32).buffer;
  }

  global.AudioUtils = { floatToInt16, floatToInt16Buffer };
})(window);
