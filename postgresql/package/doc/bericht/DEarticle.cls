\NeedsTeXFormat{LaTeX2e}\relax
\ProvidesClass{DEarticle}[2007/07/26 DE erweiterte Article Class]
\DeclareOption{a4paper}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{a4paper}
\ProcessOptions\relax
\LoadClass{article}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[ngerman]{babel}
\endinput
