## Add new application
```bash
mkdir applications/<lib-name>
uv init applications/<lib-name>
```

## Add application to external project file
```bash
uv add "<lib-name> @ git+https://github.com/gruenlab/bioinfo-tools.git@v0.1.0#subdirectory=applications/<lib-name>"
```