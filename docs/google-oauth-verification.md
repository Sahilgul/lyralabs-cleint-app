# Google OAuth verification — submit on day 1

This is the #1 launch blocker per the plan. Sensitive scopes (Drive, Sheets, Calendar, Docs, Slides) require Google's OAuth verification process, which can take 4–6 weeks and may require a CASA Tier 2 security audit ($1,500–$15,000 with a third-party assessor).

## Steps

1. **Set up your Google Cloud project** (the same project that owns your OAuth client):
   - Console → APIs & Services → OAuth consent screen
   - Set User Type to **External** (Internal won't let public users install the app)
   - Fill in: app name, user support email, app logo (120×120 PNG), homepage URL, privacy policy URL, terms of service URL, authorized domains, developer contact
2. **Enable required APIs** for the project:
   - Google Drive API, Google Docs API, Google Sheets API, Google Calendar API, Google Slides API
3. **Add your scopes** (must match exactly what `GOOGLE_OAUTH_SCOPES` requests):
   - `.../auth/drive` (sensitive)
   - `.../auth/documents` (sensitive)
   - `.../auth/spreadsheets` (sensitive)
   - `.../auth/calendar` (sensitive)
   - `.../auth/presentations` (sensitive)
4. **Prepare verification assets**:
   - YouTube demo video showing each sensitive scope in action with a real Google account (the in-app consent screen, the action being performed)
   - Public privacy policy that explicitly mentions Google user data, retention, and third-party processors
   - Domain verification (Search Console)
5. **Submit for verification** from the OAuth consent screen page
6. **CASA assessment** — only required for *restricted* scopes; `auth/drive` is restricted, so be ready
7. **Test mode in the meantime**: while in test mode, add up to 100 test users (your design partners) — this unblocks paid pilots before verification clears

## When you're blocked

- Use `https://www.googleapis.com/auth/drive.file` (per-file, non-restricted) instead of `auth/drive` for any tools that don't *need* full Drive listing. Cuts the verification scope significantly.

## Tracker

- [ ] Cloud project created
- [ ] OAuth consent screen configured (External)
- [ ] APIs enabled
- [ ] Scopes added
- [ ] Privacy policy + ToS published
- [ ] Domain verified in Search Console
- [ ] Demo video recorded
- [ ] Verification submitted (date: ___)
- [ ] CASA assessment scheduled (if needed)
- [ ] Verification approved (date: ___)
