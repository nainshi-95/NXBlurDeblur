  // inter recon
  xDecodeInterTexture(cu);

#if ENABLE_NNLF
  if (cu.cs->sps->m_nnlfStore && !cu.cs->parent)
  {
    if (cu.rootCbf)
    {
      cu.cs->getPredBufCustom(cu).copyFrom(cu.cs->getPredBuf(cu));
    }
    else
    {
      cu.cs->getPredBufCustom(cu).copyClip(cu.cs->getPredBuf(cu), cu.cs->slice->m_clpRngs);
    }
  }
#endif

  DTRACE(g_trace_ctx, D_TMP, "reco ");
  DTRACE_CRC(g_trace_ctx, D_TMP, *cu.cs, cu.cs->getRecoBuf(cu), &cu.Y());

#if DECODER_STORE_INTER_LUMA_SAMPLES
  xStoreInterLumaSamples(cu);
#endif
  cu.cs->setDecomp(cu);
}

#if DECODER_STORE_INTER_LUMA_SAMPLES
void DecCu::xStoreInterLumaSamples(const CodingUnit &cu)
{
  m_interLumaPredSamples.clear();
  m_interLumaReconSamples.clear();
  m_interLumaPredWidth  = 0;
  m_interLumaPredHeight = 0;
  m_interLumaReconWidth = 0;
  m_interLumaReconHeight = 0;

  if (!cu.Y().valid())
  {
    return;
  }

  const CompArea &area    = cu.Y();
  const CPelBuf   predBuf = cu.cs->getPredBuf(area);
  const CPelBuf   recoBuf = cu.cs->getRecoBuf(area);
  const CPelBuf   recoPic = cu.cs->picture->getRecoBuf(COMP_Y);

  m_interLumaPredWidth  = area.width;
  m_interLumaPredHeight = area.height;
  m_interLumaReconWidth = area.width + 3;
  m_interLumaReconHeight = area.height + 3;

  m_interLumaPredSamples.reserve(area.width * area.height);
  for (int y = 0; y < area.height; ++y)
  {
    for (int x = 0; x < area.width; ++x)
    {
      m_interLumaPredSamples.push_back(predBuf.at(x, y));
    }
  }

  m_interLumaReconSamples.reserve((area.width + 3) * (area.height + 3));

  auto clampToPicture = [&](int x, int y) -> Pel
  {
    x = std::max(0, std::min(x, recoPic.width - 1));
    y = std::max(0, std::min(y, recoPic.height - 1));
    return recoPic.at(x, y);
  };

  for (int y = -3; y < area.height; ++y)
  {
    for (int x = -3; x < area.width; ++x)
    {
      if (x >= 0 && y >= 0)
      {
        m_interLumaReconSamples.push_back(recoBuf.at(x, y));
      }
      else
      {
        m_interLumaReconSamples.push_back(clampToPicture(area.x + x, area.y + y));
      }
    }
  }
}
#endif
