fs::path getDatasetDumpRoot(const EncCfg *encCfg)
{
  if (encCfg != nullptr && !encCfg->m_trainingDumpDir.empty())
  {
    return fs::path(encCfg->m_trainingDumpDir);
  }
  return fs::current_path() / "dataset_dump";
}
