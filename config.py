"""
config.py - Centralized Developer and Application Configuration.

This module centralizes all personal branding, institutional affiliations,
contact URLs, and project configuration constants. Modify this single file
to update developer information throughout the entire application.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeveloperConfig:
    """Developer personal branding and contact metadata."""
    full_name: str = "Sarkar Tuhassinul Arnob"
    designation: str = "ICT Student and AI/ML Project Developer"
    department: str = "Department of ICT"
    university: str = "Islamic University, Bangladesh"
    email: str = "arnob.ict.iu@gmail.com"
    phone: str = "+8801811860339"
    whatsapp_url: str = "https://wa.me/8801811860339"
    github_url: str = "https://github.com/TuhassinulArnob"
    linkedin_url: str = "https://www.linkedin.com/in/tuhassinularnob/"
    repo_url: str = "https://github.com/TuhassinulArnob/HandWriting_Detection_simpleANN"
    # Portfolio URL is intentionally None until configured by the developer
    portfolio_url: Optional[str] = None

    @property
    def email_mailto(self) -> str:
        """Formatted mailto URL for direct email client opening."""
        return f"mailto:{self.email}"

    @property
    def affiliation_line(self) -> str:
        """Complete academic affiliation string."""
        return f"{self.department}, {self.university}"

    @property
    def has_portfolio(self) -> bool:
        """Indicates whether a valid portfolio website URL has been provided."""
        return bool(self.portfolio_url and self.portfolio_url.strip())


@dataclass(frozen=True)
class AppConfig:
    """Application title, branding, and metadata."""
    app_title: str = "Handwritten Digit Recognition"
    app_subtitle: str = "Recognizing handwritten digits using an Artificial Neural Network."
    app_tagline: str = (
        "An educational, end-to-end Machine Learning web application demonstrating digit "
        "classification (0–9) using a traditional Multilayer Perceptron (MLP) trained on the MNIST dataset."
    )
    version: str = "2.0.0"
    repo_url: str = "https://github.com/TuhassinulArnob/HandWriting_Detection_simpleANN"


# Singleton instances for global import
DEVELOPER = DeveloperConfig()
APP_CONFIG = AppConfig()
