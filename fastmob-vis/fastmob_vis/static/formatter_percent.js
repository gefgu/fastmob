  function formatPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '';
    return number.toFixed(2) + '%';
  }
